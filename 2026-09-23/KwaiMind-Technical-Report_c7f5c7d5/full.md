# KwaiMind Technical Report

KwaiMind Team, Kuaishou Group

<sup>§</sup> https://github.com/KwaiMmu/KwaiMind

Instruction-based image editing has made rapid progress, yet commercial content production demands more than general instruction following and visual quality. Generated images must preserve product identity, render promotional text accurately, and appeal to users. We present KwaiMind, an image editing system that combines broad editing competence with domain-specific capabilities for e-commerce. KwaiMind integrates an agent-based data engine, a staged adaptation pipeline, and a commercial benchmark. The data engine coordinates filtering, targeted generation, hierarchical annotation, and quality auditing to maintain approximately 1.8 million high-quality editing pairs. Using a multimodal difusion transformer (MMDiT), we perform continued pre-training and supervised fine-tuning on general and e-commerce data, followed by preference optimization and online reinforcement learning. A general-purpose vision-language judge is complemented by specialized rewards for click-through rate (CTR), text rendering, and fine-grained product consistency. We optimize these objectives through specialized policies and consolidate their capabilities into a single editor via on-policy distillation. To assess practical utility, we introduce Ecom-Bench, covering 11 commercial editing tasks with task-specific visual evaluation and CTR-based ranking. KwaiMind achieves the strongest overall scores among the evaluated open-source editors on ImgEdit, GEdit, both language splits of REDEdit, and Ecom-Bench visual quality, while attaining the highest aggregate CTR ranking score among the compared systems. Ofline, CTR-guided optimization increases the proportion of generated images whose predicted CTR exceeds that of the original product image from 12.16% to 37.41%. In an online A/B experiment, CTR-based selection of product main images yields an approximately 2.44% relative increase in actual CTR. These results demonstrate the value of combining domain-specific data, reward-driven alignment, and commercially grounded evaluation for practical image editing.

![](images/68924c2d4d4a630d385fc8fbaf7306e959622cc17ca66d0aafe3314077d7db1f.jpg)

![](images/fef9935f99ef60dc322ce49eabb0e88ecdda51fad3204090bcb809c6fb3d2c8c.jpg)  
Figure 1 | Overall comparison on Ecom-Bench visual quality and Ecom-CTR ranking. Hatched bars denote closed-source models.

## Contents

1 Introduction 4   
2 Data 6   
2.1 Data Agent 6   
2.2 Filter Agent 7   
2.3 Generation Agent 8   
2.4 Caption Agent . 9   
2.5 Coordinator Agent 9   
2.6 End-to-End Data Pipeline 10   
2.7 Dataset Scale and Final Composition 11   
3 Training 13   
3.1 Architecture . 13   
3.2 Continued Pre-Training . 14   
3.3 Supervised Fine-Tuning 14   
3.4 Reinforcement Learning with Human Feedback 15   
4 Benchmark 22   
4.1 General Image Editing Benchmark 22   
4.2 Ecom-Bench: E-commerce Image Editing Benchmark 23   
5 Evaluation 26   
5.1 General Image Editing Results . 26   
5.2 Ecom-Bench Results 28   
5.3 Visualization 29   
6 Conclusion 35

![](images/3edf25c1ecef70db7500bcde4e567780e3c7815b394a1bfcb84022959c9de072.jpg)  
Figure 2 | Showcases of KwaiMind across multiple image-editing tasks.

![](images/61f9aef5cd2861e1a7ef422d3df4cd91bed37aff631e8df96f70775d92cd819f.jpg)

![](images/c3646af21d56a0b3a694eb32154fd49fa6b8be6cd2400d6ac7bdd6b985573e95.jpg)

![](images/fb37c0379515fa9612bdf0d073103534b72092db104f57bc8645153a2f95e2d4.jpg)

![](images/5408e6c5182fad274dfc4999c886ee07337052a55eee56bc3511504089f0148f.jpg)  
Figure 3 | Overall comparison on general image editing benchmarks: ImgEdit, GEdit, and the English and Chinese splits of REDEdit. Hatched bars denote closed-source models.

## 1. Introduction

Instruction-based image editing has advanced rapidly, and modern difusion editors can now follow open-domain natural-language instructions to add, remove, replace, restyle, and recompose visual content with high fidelity (Brooks et al., 2023; Labs et al., 2025; Wei et al., 2025; Wu et al., 2025a; Yu et al., 2025). Difusion models also have broad applications in other domains (Chang et al., 2026; Gong et al., 2026; Li et al., 2025; Lin et al., 2026; Xia et al., 2025b, 2026), while e-commerce image production ofers a particularly valuable application of instruction-driven editing. On a modern e-commerce platform, every product may require many display images: a garment shown on diferent models and in diferent poses, a product placed against a variety of scenes, a promotional poster that highlights selling points, or a clean cutout for a catalog. Producing these images by hand is slow and expensive, and the volume grows with the size of the catalog, which makes automated, instruction-driven editing an attractive and valuable tool for merchants (Wang et al., 2021; Yang et al., 2024).

Despite this promise, general-purpose editors do not directly meet the demands of commercial image production. E-commerce editing is governed by requirements that generic benchmarks neither isolate nor reward. The product itself must remain faithful across a large change of background, pose, or viewpoint, since any drift in shape, color, texture, or brand identity misrepresents the item being sold (Li et al., 2026b). Text on product images, including prices, promotions, and selling points, must be rendered legibly and correctly, and even small glyph-level errors are immediately visible to shoppers (Tuo et al., 2024; Wang et al., 2026a). Garment tasks such as virtual try-on impose their own constraints on fit, drape, and material fidelity, and they require reasoning jointly over more than one reference image (Choi et al., 2024). Above all, the ultimate measure of a commercial image is whether it attracts users, a property that is not fully captured by generic perceptual quality and that a general editor is not trained to optimize (Wang et al., 2021; Yang et al., 2024). These requirements are largely absent from the data, the objectives, and the evaluation protocols on which general editing models are built.

We identify three obstacles that stand between a capable general editor and a production-grade e-commerce editing system. First, high-quality training data for the domain is scarce: e-commerce editing spans many specialized tasks, real product photography is noisy and unevenly distributed across categories, and constructing clean, instruction-aligned editing pairs at scale requires filtering, generation, and annotation far beyond what a fixed public corpus provides (Wei et al., 2025; Yu et al., 2025). Second, standard training objectives do not target the domain-critical properties above. Supervised fine-tuning on imitation data teaches instruction following but does not directly optimize product identity preservation, text correctness, or commercial appeal. Third, existing benchmarks measure broad editing competence rather than the requirements of commercial image production, making it dificult to assess a model’s suitability for real-world e-commerce workflows.

Addressing these challenges requires coordinated design across data construction, model alignment, and evaluation. We therefore introduce KwaiMind, an instruction-based image editing system for e-commerce that brings these elements together through three tightly connected components.

An agent-based data engine. We construct training data with a multi-agent pipeline that turns raw and generated imagery into clean, instruction-aligned editing pairs at scale (§2). A Filter Agent enforces data quality through an online pre-filter over intrinsic image properties and a simulatedhuman post-filter that approximates human review, a Generation Agent repairs rejected samples and synthesizes data for under-covered tasks, a Caption Agent produces hierarchical instructions and verifies them through a reverse-audit loop, and a Coordinator Agent schedules the whole process as an auditable, closed-loop state machine with human intervention reserved for high-value and boundary cases. The engine produces domain-concentrated, quality-controlled data across the full range of e-commerce editing tasks.

A domain-adapted model with reward-driven alignment. Following the architectural design of Qwen-Image (Wu et al., 2025a), we build on a Multimodal Difusion Transformer (MMDiT) (Esser et al., 2024) and adapt the model to the e-commerce domain through a staged pipeline of continued pre-training, supervised fine-tuning, and reinforcement learning from human feedback (§3). The Data Agent maintains approximately 1.8M high-quality training pairs spanning general and e-commerce editing. Continued pre-training uses the subset above 720p, together with instruction augmentation, to develop high-resolution editing capabilities across both domains. Supervised fine-tuning then refines editing precision on a task-balanced corpus curated by the Data Agent, comprising 115.0K general editing pairs and 119.9K e-commerce pairs selected under stricter criteria for instruction accuracy, content preservation, and visual quality. The alignment stage is where the domain-critical objectives are optimized directly. We first apply Direct Preference Optimization (Rafailov et al., 2023; Wallace et al., 2024) in a mixed ofline and online regime, then run online reinforcement learning on the forward difusion process, driven by a vision-language judge for general edits and by three dedicated reward models that target the properties general rewards miss: a click-through rate model for commercial appeal, a coarse-to-fine reward for visual text rendering, and a consistency reward for product and model identity preservation Li et al. (2026b); Wang et al. (2026c). Because each objective is optimized most efectively on its own, we finally consolidate the specialized policies into a single model through on-policy distillation, yielding one editor that inherits all capabilities without cross-task interference.

A commercial-grade benchmark. We introduce Ecom-Bench, an evaluation suite purpose-built for e-commerce image editing (§4). It covers 11 representative tasks spanning garment and wearable editing, composition and layout, text operations, and appearance transfer, and it scores each task with a behavior-anchored, per-task selection of general and e-commerce-specific dimensions under a geometric-mean protocol that penalizes any single critical defect. Beyond judge-based scores, Ecom-Bench reports a learned click-through rate score that estimates the commercial attractiveness of a generated image. Alongside Ecom-Bench, we evaluate on established general-domain image editing benchmarks.

The main contributions of this work are as follows.

• We present an agent-based data engine that automates filtering, targeted generation, hierarchical captioning, and coordination into a closed-loop system for producing high-quality e-commerce editing data at scale.

• We adapt a strong open-source editor to the e-commerce domain with an alignment stage that combines mixed ofline and online preference optimization, forward-process reinforcement learning, and three e-commerce reward models for commercial appeal, text rendering, and identity consistency, consolidated into a single model by on-policy distillation.

• We introduce Ecom-Bench, an 11-task benchmark with domain-tailored, behavior-anchored metrics and a learned click-through rate score, providing an evaluation protocol aligned with the requirements of commercial image production.

## 2. Data

This section will describe the agent-based system for filtering, generating, annotating, and coordinating training data.

## 2.1. Data Agent

Data Agent is a multi-agent collaborative system for producing image-editing data. Coordinated by Coordinator Agent and jointly executed with the Filter, Generation, and Caption Agents, with human reviewers serving as a fallback at critical checkpoints, it transforms raw materials end-to-end into high-quality source-target-caption triplets, while simultaneously producing the evaluator dataset used to iterate the simulated-human VLM evaluator.

## 2.1.1. System Architecture

The system comprises four collaborating sub-agents. All Skills are uniformly registered in the Skill Registry for dynamic invocation, and human reviewers are explicitly involved at critical checkpoints to handle boundary and failure cases.

Coordinator Agent. The global scheduling hub. It maintains the sample lifecycle through a finite state machine, drives the sub-agents with standardized work orders, and manages loop budgets, gap-triggered supplementary generation, and the accumulation of strategy experience.

Filter Agent. The data quality gatekeeper. Pre-Filter removes low-quality and non-compliant samples before annotation under two collaboration modes, Voter and Aspect. Post-Filter, applied after annotation, scores each sample along three criteria with a simulated-human VLM evaluator that iterates on itself.

Generation Agent. The data remediation and completion component. The Modification branch repairs recoverable samples based on diagnostic reports, and the Supplement branch fills subtask gaps. An LLM Planner performs task parsing and Skill routing, and the actual generation is carried out by four categories of expert models.

Caption Agent. The textual annotation component. It produces three-tiered hierarchical captions for each sample, uses diference-mask guidance to direct the VLM toward the edited regions, and verifies annotation quality through a closed-loop reverse audit performed by an LLM.

## 2.2. Filter Agent

Filter Agent serves as the core component responsible for data filtering and quality control. It is divided into two stages according to its position in the pipeline. Pre-Filter focuses on intrinsic image quality, removing low-quality, non-compliant, or duplicate samples before images enter the annotation stage. Post-Filter simulates human review by performing fine-grained evaluation on data tuples with generated captions, approximating human review capability and continuously accumulating training data for downstream model iteration.

## 2.2.1. Pre-Filter

Multi-Model Collaboration. Each Pre-Filter Skill uses either Voter or Aspect collaboration. Voter mode handles holistic judgments such as AIGC and anime detection or content compliance. It invokes � heterogeneous VLMs in parallel and adopts the majority verdict; when no majority exists, a Judge model aggregates their verdicts and rationales Wang et al. (2022); Zheng et al. (2023). Aspect mode handles composite criteria such as perceptual quality, aesthetics, and product-task consistency. It decomposes each criterion into � approximately orthogonal aspects evaluated by independent prompts or dedicated detectors, followed by weighted aggregation with a Judge model Song et al. (2024). All Skills output an evidence chain and one of four verdicts: Passed, Rejected Recoverable, Rejected-Unrecoverable, or Uncertain. These verdicts map to the PreFilter-{Passed, Recoverable, Rejected, Uncertain} states. At the Pre-Filter stage, Uncertain denotes voter disagreement, contradictory aspect evidence, or low confidence.

Pre-Filter Skills. We implement four Pre-Filter Skills. Deduplication, basic visual feature filtering, consistency filtering, and the perceptual quality and aesthetics sub-Skill use Aspect mode. The AIGC and anime detection and content compliance sub-Skills use Voter mode.

1. Deduplication. Global deduplication uses CLIP embeddings Radford et al. (2021) for nearestneighbor retrieval and clustering. Edit-pair deduplication combines CLIP similarity with PSNR, SSIM, and LPIPS Zhang et al. (2018) to remove near-identity pairs Team et al. (2026).

2. Basic visual feature filtering. Thresholds on saturation, brightness, and RGB entropy remove under-exposed, over-exposed, color-distorted, and near-uniform images Team et al. (2026). For e-commerce data, CTR signals further exclude images with poor historical performance.

3. Compliance filtering. This Skill comprises three sub-Skills. (1) AIGC and anime detection excludes images with a pronounced AI-generated appearance and anime-style images. (2) Perceptual quality and aesthetics assessment evaluates blur, noise, exposure, physical anomalies, composition, color harmony, watermarks, and text overlays. (3) Content compliance review uses multi-model voting to detect nudity, graphic violence, and other prohibited content and produces an auditable rejection report.

4. Consistency filtering. For editing pairs involving products, persons, and related subjects, this Skill checks category, subject identity, and core visual attributes between source and target images.

Pre-Filter Output and Routing. Each verdict includes routing metadata. Passed records confidence; Rejected-Recoverable records rejection reasons, recovery hints, and evidence chains; Rejected-Unrecoverable records rejection categories; and Uncertain records the disagreement source and aggregate confidence. Coordinator routes samples using these fields. Batch-level reports record pass rates, Skill-level rejection distributions, aspect scores, collaboration disagreement, and sampled positive and negative cases for data monitoring and Skill updates.

## 2.2.2. Post-Filter

Evaluation Dimensions. Post-Filter applies a simulated-human VLM evaluator to captioned tuples of a source image, target image, and caption. Following FireRed Team et al. (2026), it evaluates three dimensions. (1) Instruction consistency measures whether the source-to-target change correctly and completely follows the caption without extraneous edits. (2) Edit consistency measures preservation outside the intended edit region, including subject identity. (3) Perceptual quality measures sharpness, artifacts, and aesthetics. For each dimension, the evaluator outputs a score on a 1–10 scale, together with confidence and a rationale. It then assigns Accepted, Rejected, Uncertain, or OOD, corresponding to the PostFilter-{Accepted, Rejected, Uncertain, OOD} states. Uncertain denotes an in-distribution sample with contradictory dimension scores or confidence below a preset threshold. OOD denotes an input outside the evaluator’s supported distribution. Both states are routed to human review.

Evaluator Training Data. Evaluator training data comes from four sources. Positive samples are high-confidence Accepted outputs from the current evaluator. Auto-constructed negatives are generated by an LLM through controlled perturbations of attributes, subjects, operations, or categories in Accepted samples Team et al. (2026). Human samples include reviewed contrastive pairs and manually corrupted samples. Pre-Filter negatives are high-confidence, low-disagreement samples rejected by Pre-Filter in Section 2.2.1. The evaluator is updated through prompt revision with highvalue errors and accumulated experience as few-shot examples, and through supervised fine-tuning. Updates are manually deployed when the accumulated samples reach a preset threshold and ofline validation shows no performance degradation.

## 2.3. Generation Agent

Generation Agent performs Modification and Supplement. Modification repairs samples marked as Rejected-Recoverable by Pre-Filter, while Supplement fills subtask gaps detected by Coordinator Agent. In both branches, an LLM Planner routes each work order to an expert model or API. The expert produces multiple candidates, from which the highest-scoring result is retained. Each output records its execution path, expert models, intermediate boxes or masks, and prompts. Modified samples enter the Generated-Modified state with a region mask for local Pre-Filter, whereas supplementary samples enter the Generated-Supplement state and undergo full-image Pre-Filter. Modification is limited to � rounds, and samples exceeding this budget are routed to the human review queue.

## 2.3.1. LLM Planner

The LLM Planner converts each work order into an executable plan through three steps: (1) Diagnostic parsing. For Modification, it extracts repair targets from the rejection reasons, recovery hints, and evidence chain, including missing subjects, attribute mismatches, and unintended edits. For Supplement, it determines the target subtask, scene constraints, and diversity requirements from the gap list and dataset distribution. This step is omitted when the work order already specifies the target scene. (2) Skill routing. The Planner inserts localization or segmentation before spatially constrained tasks such as replace, remove, and change-background. It then selects a Skill according to task type, scene complexity, and the Skill capability profile. Expert models remain fixed within each Skill and are not exposed to the Planner. (3) Prompt generation. The Planner generates semantically equivalent and lexically diverse prompts for the selected Skill to reduce mode collapse Brooks et al. (2023).

## 2.3.2. Expert Execution

For composite edits, the Skill layer follows the Task Splitting mechanism in FireRed Team et al. (2026). A VLM decomposes each instruction into ordered atomic operations, and the Router sequentially invokes the corresponding experts to reduce per-step complexity and preserve structure. We group these experts by control signal: (1) Instruction-driven editing. FLUX.2 Black Forest Labs (2025) and Qwen-Image-Edit-2511 Wu et al. (2025a) handle general edits, LongCat-Image Meituan LongCat Team et al. (2025) handles dense Chinese text, and Seedream and NanoBanana-2 handle complex multimodal instructions. (2) Perception. GroundingDINO Liu et al. (2024), SAM2 Ravi et al. (2025), and RMBG-2.0 Zheng et al. (2024) provide bounding boxes, masks, and foreground separation for spatially constrained edits. (3) Structured-control editing. Mask-conditioned FLUX.2 and SDXL-Inpainting Podell et al. (2024) perform localized edits, while DWPose Yang et al. (2023b) provides keypoints for pose transfer. (4) Deterministic synthesis. We use 3D parametric templates, structured layout templates, and deterministic image-processing operators Team et al. (2026) for color transfer, sharpening, layout control, and parametric pose or expression control. This group also serves as a fallback and a supplementary data source.

## 2.4. Caption Agent

Caption Agent annotates images and editing pairs marked as Passed by Pre-Filter through hierarchical captioning, mask-guided prompting, and bounded reverse auditing. The VLM produces three captions for each sample: (1) Detailed caption. It describes the image content for text-to-image data or the attribute, spatial, and semantic changes in an editing pair. (2) Concise caption. An LLM compresses the detailed caption using randomly sampled syntactic structures from multiple VLMs and lexicons to reduce template bias Singla et al. (2024). (3) Simulated-user caption. The LLM rewrites the concise caption as a colloquial, help-seeking instruction. For editing pairs, we compute an approximate change mask from the pixel-level diference between the source and target images. The mask is overlaid on the target image and encoded as Set-of-Mark boxes and indices in the prompt Yang et al. (2023a). This directs the VLM to describe edited regions, including small object replacements, local tagline removal, and text modification.

After captioning, a reverse audit checks semantic accuracy following a generate-then-verify procedure Wu et al. (2025c). Given only the caption, an LLM infers the expected visual content and compares its key slots with the known metadata. Any mismatch is added to the prompt for caption regeneration. The loop terminates when the audit passes or reaches � rounds Madaan et al. (2023); remaining failures are forwarded by the Coordinator for human review.

## 2.5. Coordinator Agent

Coordinator Agent maintains the global dataset state, schedules sample batches, and dispatches standardized work orders across sub-agents. It also manages loop budgets and human-review entry points. Each work order records its branch triggers for reproducibility and auditing.

## 2.5.1. State Machine and Work Orders

Coordinator tracks each sample through the finite state machine in Table 1. Every state transition emits a work order containing the source and target states, rejection reason, evidence chain, execution path, and metadata. The pipeline contains two feedback loops: Pre-Filter with Modification and Caption with Reverse-Audit. Both loops are capped at � rounds. Samples exceeding the loop budget enter the corresponding stage-specific Escalated state and are routed to human review.

Table 1 | Sample states maintained by Coordinator Agent. Each state determines the next work-order destination. States ending in Escalated indicate that the corresponding loop budget has been exhausted and human review is required.
<table><tr><td>State</td><td>Description</td></tr><tr><td>Ingested</td><td>Newly ingested sample</td></tr><tr><td>PreFilter-{Passed, Recoverable, Rejected, Uncertain}</td><td>Pre-Filter verdict</td></tr><tr><td>Generated-{Modified, Supplement}</td><td>Generation output</td></tr><tr><td>Captioned</td><td>Captioning completed</td></tr><tr><td>Modification-Escalated</td><td>Modification budget exhausted</td></tr><tr><td>Caption-Escalated</td><td>Reverse-audit budget exhausted</td></tr><tr><td>PostFilter-{Accepted, Rejected, Uncertain, OOD}</td><td>Post-Filter verdict</td></tr><tr><td>Evaluator-{Human, Negative, Positive}</td><td>Evaluator training data</td></tr><tr><td>Human-{Approved, Rejected, Recoverable, Relabeled}</td><td>Human verdict</td></tr><tr><td>Ingested-To-Trainset</td><td>Pending database entry</td></tr></table>

## 2.5.2. Scheduling Memory

Coordinator maintains loop memory and an experience bufer. Loop memory records each sample’s iteration count and rejection sequence for budget control and stopping. The experience bufer stores representative successes and failures, human verdicts, and automatically constructed negatives for Skill prompt updates and SFT training. This design combines per-sample loop state with cross-batch experience Zhang et al. (2025). For Supplement, Coordinator periodically checks subtask coverage, category distribution, and diversity against preset thresholds. Detected gaps trigger Supplement work orders to Generation Agent. All scheduling decisions follow a work-order-based ReAct loop Yao et al. (2022).

## 2.5.3. Human-in-the-Loop

Human intervention occurs at three entry points: (1) Milestone review. After each high-cost stage, Coordinator reports the batch pass rate, Skill-level rejection distribution, subtask-level pass rate, and disagreement by collaboration mode. Representative accepted, rejected, and boundary samples are included for approval before the next stage. (2) Exception handling. Uncertain samples from either filtering stage, Post-Filter OOD samples, loop-budget overflows, and operational threshold violations are routed to human review. Reviewers resolve these cases and annotate boundary samples. (3) Iteration decisions. Reviewers select samples for the experience bufer, Skill prompt updates, or SFT training. They also trigger retraining and deployment of the simulated-human VLM evaluator.

## 2.6. End-to-End Data Pipeline

Figure 4 summarizes the end-to-end pipeline. Coordinator uses its state machine to route sample batches through Pre-Filter, Generation, Caption, and Post-Filter, with the Skill Registry providing shared capabilities. The pipeline produces the main training set and an evaluator dataset for iterative VLM evaluator updates.

Pre-Filter routes Passed samples to Caption Agent, Rejected-Recoverable samples with diagnostic reports to Modification, and Uncertain samples to human review. Rejected-Unrecoverable samples are archived, with high-confidence cases added to the evaluator dataset. Supplement addresses subtask gaps detected by Coordinator, and all Generation outputs return to Pre-Filter. Caption Agent applies a bounded generation, reverse-audit, and revision loop before qualified source-image, targetimage, and caption triplets enter Post-Filter. High-confidence Accepted samples enter the training-set ingestion queue, while high-confidence Rejected samples and automatically perturbed hard negatives are archived as evaluator negatives. Post-Filter Uncertain and OOD samples are routed to human review. Accepted and rejected outputs also update the evaluator dataset. Coordinator enforces all loop budgets and routes milestone reports, exceptional or low-confidence samples, and on-demand inspections to human review. Human verdicts update pipeline states and evaluator data.

![](images/f9b82579e63f5c0b8465674e4a4dea5eff0eed594a9f6f3348d5dc9346ed6196.jpg)  
Figure 4 | Overview of the end-to-end data pipeline. Coordinator Agent routes samples through filtering, generation, captioning, and post-filtering with bounded feedback loops and human review.

## 2.7. Dataset Scale and Final Composition

We apply the Data Agent to a mixture of public editing datasets and proprietary Kwai pairs. These sources contain approximately 26.2M source–target examples before the final quality-control and deduplication stages. Through filtering, targeted modification of recoverable samples, and gap-driven supplementary generation, the Data Agent maintains approximately 1.8M high-quality training pairs for model development.

Candidate editing pairs that pass Pre-Filter and complete captioning and reverse auditing enter Post-Filter, where instruction consistency, edit consistency, and perceptual quality are each scored on a 1–10 scale as defined in §2.2.2. Following the routing in §2.6, high-confidence Accepted triplets enter the training-set ingestion queue, while high-confidence Rejected samples are archived as evaluator negatives. The maintained training pairs are then selected according to their downstream role: broad high-resolution pairs support CT, while more selective and task-balanced subsets support SFT and final alignment. Modification repairs samples identified as recoverable by Pre-Filter; Supplement addresses task or visual-pattern gaps detected by Coordinator. Outputs from both branches return to Pre-Filter and proceed through captioning and Post-Filter before they can enter the training corpus.

![](images/73f9ea48a66901a2dc2b244be7898bdfef13db31994808c58f33026de28e2c01.jpg)  
Figure 5 | Composition of the curated SFT corpus. The upper half shows the e-commerce subset and the lower half shows the general editing subset; surrounding examples illustrate representative source–target pairs from diferent tasks.

To prepare the data for supervised fine-tuning, we organize candidates from the maintained 1.8M-pair corpus by general and e-commerce editing tasks. We first filter candidate pairs using their recorded Post-Filter scores, then apply an additional round of task-specific screening with criteria tailored to each subtask across instruction accuracy, content preservation, and visual quality, alongside task balancing and quality auditing. This process yields 234.9K SFT pairs, comprising 115.0K general editing pairs covering atomic and compositional edits and 119.9K e-commerce pairs drawn from proprietary data collected from real merchant demands. The e-commerce subset spans 11 tasks grouped into three families: Garment & Wearable, Composition & Layout, and Text & Marketing. Figure 5 shows the task distribution of the curated SFT corpus, and the detailed selection criteria are provided in §3.3.

Finally, we construct dedicated preference and reward-training subsets from the maintained corpus. We stratify e-commerce and general editing examples across the target tasks, while using additional expert-generated pairs when required to improve long-tail coverage. This staged reuse of the same quality-controlled pool supports CT, SFT, and final alignment without treating earlier filtered data as discarded.

Table 2 | Data sources processed by the Data Agent and the resulting maintained training corpus.
<table><tr><td>Data source</td><td>Number of samples</td></tr><tr><td>ScaleEdit-12M (Chen et al., 2026) X2Edit (Ma et al., 2026) AnyEdit (Yu et al., 2025) KwaiData</td><td>12.0M editing pairs 3.7M editing pairs 2.5M editing pairs 8.0M editing pairs</td></tr><tr><td>Pair-based source total Maintained training corpus</td><td>26.2M pairs ~1.8M pairs</td></tr></table>

## 3. Training

We build our editing model on top of a strong open-source foundation Wu et al. (2025a) and adapt it through a staged pipeline comprising continued pre-training (CT), supervised fine-tuning (SFT), and reinforcement learning from human feedback (Wu et al., 2026). CT combines instruction augmentation with a broad mixture of general and e-commerce editing pairs above 720p, prepared by the Data Agent, to develop high-resolution editing capabilities across both domains. SFT then refines instruction accuracy, content preservation, and visual quality using a task-balanced subset curated through score-based filtering and an additional round of screening with task-specific criteria across these three dimensions. The final alignment stage combines preference optimization, online reinforcement learning driven by task-specific reward models, and multi-task consolidation.

## 3.1. Architecture

Our architecture follows Qwen-Image-Edit-2511 (Wu et al., 2025a) and is based on a double-stream Multimodal Difusion Transformer (MMDiT) (Esser et al., 2024). Three input streams are concatenated into a single token sequence for dense bidirectional attention: latent tokens of the target image produced by a variational autoencoder (VAE) encoder (Rombach et al., 2022), latent tokens of one or more reference images, and textual instruction embeddings produced by a vision-language encoder. Jointly processing the reference and target in a single stream enables each target token to attend to the reference content at every layer, a capability essential for preserving identity and layout in the editing tasks that predominate in commercial image production.

Position information follows a unified rotary scheme. Reference and target image tokens share a common spatial coordinate grid and are distinguished by a temporal ofset, so that spatially corresponding regions of the reference and the edited result are placed in register while remaining separable by the model. Clean reference latents and noised target latents receive distinct time conditioning, which prevents the model from confusing the fixed reference with the signal it must denoise.

Because e-commerce editing frequently requires more than one reference, for example a model image together with a garment image in virtual try-on, the input stream accepts a variable number of reference images. Multiple references are encoded independently by the VAE encoder and appended to the sequence, each carrying its own positional and temporal tags. This paired multi-reference format is used consistently across training and inference so that the model learns to bind attributes from the correct source image.

## 3.2. Continued Pre-Training

Continued pre-training (CT) adapts the foundation model using a broad mixture of general and e-commerce editing pairs from the approximately 1.8M maintained training pairs described in §2.7. These data are obtained through the Data Agent described in §2.1. From this automatically curated pool, we retain only image pairs with resolutions above 720p for CT, adapting the model to high-resolution editing scenarios where fine-grained textures, product details, and scene structure need to remain clear and consistent. Training on this high-resolution subset combines instruction augmentation with exposure to editing operations across both domains. E-commerce data is included from the outset, allowing the model to learn from diverse e-commerce products, materials, presentation scenes, and operation types alongside general editing tasks. This stage establishes broad, highresolution editing competence across domains, providing the foundation for the more selective SFT stage.

Instruction augmentation. Building on the three instruction forms produced by the Caption Agent in §2.4, we use Qwen3-VL-32B (Bai et al., 2025) to provide Chinese and English versions of each form for every source–target pair. Detailed instructions specify the editing changes and constraints, concise instructions retain the essential editing intent, and simulated-user instructions express that intent as colloquial user requests. This yields a 2 × 3 instruction set: two languages combined with three expression styles. All six variants preserve the same editing intent and share the same visual supervision. During training, we sample one language and one instruction form for each pair in each epoch, encouraging robustness to diferences in language, specificity, and phrasing rather than dependence on a single annotation style.

## 3.3. Supervised Fine-Tuning

Supervised fine-tuning (SFT) refines the CT checkpoint on the training data curated by the Data Agent in §2.7. The 234.9K pairs comprise 115.0K general editing examples and 119.9K e-commerce examples, with their task distribution illustrated in Figure 5. The general subset covers atomic and compositional editing, while the e-commerce subset includes proprietary business data collected from real merchant demands. Whereas CT emphasizes broad exposure to domains and editing operations, SFT uses this selectively curated, task-balanced corpus to improve editing precision and output quality.

Task-wise selection criteria. Following the initial filtering based on Post-Filter scores, we apply an additional task-specific filter across three dimensions: instruction accuracy, content preservation, and visual quality. The screening criteria for each dimension are tailored to the editing requirements of individual subtasks, while sharing the following general principles:

• Instruction accuracy. All requested modifications are completed, with no unintended changes beyond the instruction.

• Content preservation. Product design, color, logos, person identity, and non-edited regions remain faithful to the reference, except where a change is explicitly requested.

• Visual quality. Textures are clear, geometry is plausible, lighting and shadows are natural, and the output contains no conspicuous artifacts.

Task coverage and conditioning. The e-commerce subset spans 11 tasks grouped into Garment & Wearable, Composition & Layout, and Text & Marketing. We maintain coverage across these tasks and the general editing categories so that high-volume tasks do not dominate the curated mixture.

For tasks requiring multiple references, we retain the multi-reference input format used at inference, enabling the model to associate each reference with the appropriate subject and preserve the relevant product and identity attributes.

## 3.4. Reinforcement Learning with Human Feedback

Supervised fine-tuning teaches the model to follow editing instructions, but it optimizes a maximumlikelihood objective on curated pairs and does not directly reward the properties that determine production quality, such as instruction faithfulness, identity preservation, text legibility, and commercial appeal. We therefore add a reinforcement learning from human feedback (RLHF) stage. Following the view that denoising can be treated as a multi-step decision process amenable to policy optimization (Black et al., 2024), we align the model against explicit reward signals rather than against a fixed reference distribution alone. Related applications of task-specific reinforcement learning span robotic control (Wu et al., 2025b), multimodal reasoning and self-evolution (Heng et al., 2026; Jiang et al., 2026), and multi-view scene editing (Wang et al., 2026b).

![](images/65bf3ec27dc2c85fb928e0ab111282e4152c55e1a59d9199a1ae12db30cbdf4d.jpg)  
Figure 6 | Overview of the proposed post-training framework.

Our alignment pipeline has three parts. We first apply Direct Preference Optimization (DPO), combining an ofline phase on curated preference pairs with an online phase that regenerates pairs from the policy, to establish a stable preference-aligned checkpoint. We then run online reinforcement learning with DifusionNFT (Zheng et al., 2026), whose reward is supplied by VLMs acting as taskconditioned judges and, for the e-commerce objectives that a general judge cannot score reliably, by three dedicated reward models targeting click-through rate, visual text rendering, and detail consistency. Because each of these objectives is optimized most efectively by its own reinforcement learning run, we finally consolidate the separately trained policies into a single model with DifusionOPD (Li et al., 2026a). The remainder of this subsection describes each component.

## 3.4.1. DPO

The first alignment step is Direct Preference Optimization (Rafailov et al., 2023) applied to the difusion model (Wallace et al., 2024). We train DPO in a mixed regime that combines an ofline phase on a fixed preference corpus with an online phase that draws preference pairs from the model’s own generations. The ofline phase establishes a stable preference-aligned checkpoint from curated data, and the online phase keeps improving the policy against the distribution it actually produces, which mitigates the distribution shift that arises when a static ofline set no longer matches the evolving model.

Ofline DPO. The ofline phase trains on a fixed corpus of preference pairs. For each editing condition �, which consists of the source image and the instruction, we draw several candidate edits from the supervised checkpoint and, where available, from expert editing pipelines. Each candidate is annotated for instruction compliance and visual quality, by human reviewers for a high-precision subset and by a vision-language judge for the remainder. Within the candidates for a given condition we take a high-scoring edit as the preferred sample $x ^ { w }$ and a low-scoring edit as the rejected sample $x ^ { l }$ , so that both members of a pair share the same source and instruction and difer in editing quality. This same-condition construction isolates the quality signal from content diferences and yields the fixed corpus $\mathcal { D } _ { \mathrm { o f f } }$ . We optimize the standard Difusion-DPO Wallace et al. (2024) objective, which contrasts the current policy � against a frozen reference policy $\theta _ { \mathrm { r e f } }$ initialized from the supervised checkpoint:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { D P O } } ( \mathcal { D } ) = - \mathbb { E } _ { ( c , x ^ { w } , x ^ { l } ) \sim \mathcal { D } } \left[ \log \sigma \Big ( - \beta \big ( \Delta _ { \theta , \mathrm { r e f } } ( x ^ { w } , c ) - \Delta _ { \theta , \mathrm { r e f } } ( x ^ { l } , c ) \big ) \Big ) \right] , } \end{array}\tag{1}
$$

where $\sigma ( \cdot )$ is the logistic function, $\beta$ is the preference temperature, and $\Delta _ { \theta , \mathrm { r e f } } ( x , c ) \ = \ \mathcal { L } _ { \theta } ( x , c ) \ -$ $\mathcal { L } _ { \theta _ { \mathrm { r e f } } } ( x , c )$ is the diference between the denoising losses of the current and reference policies on sample �. Minimizing eq. (1) on $\mathcal { D } _ { \mathrm { o f f } }$ raises the relative likelihood of preferred samples over rejected ones while the reference term regularizes the update toward the supervised checkpoint.

Online DPO. Ofline DPO is bounded by its fixed corpus. As the policy improves during training, the stored pairs increasingly reflect mistakes the model no longer makes, the preferred edits fall below what the current policy can already produce, and the gradient from eq. (1) weakens. To keep the preference signal aligned with the model’s present behavior, we continue with an online phase that regenerates pairs from the policy itself. Treating denoising as a multi-step decision process and training on the model’s own samples has been shown to optimize downstream rewards more efectively than reweighting a fixed dataset (Black et al., 2024), which motivates moving from a static corpus to on-policy pairs. At each round the current sampling policy $\pi _ { \mathrm { o l d } }$ produces a group of candidate edits for a sampled condition $c ;$ each candidate is scored by the task-conditioned reward introduced in §3.4.2, that is a vision-language judge for general edits and the dedicated reward model of sections 3.4.4 to 3.4.6 for the corresponding e-commerce objective. We form an on-policy pair by taking the highest-scoring candidate as $x ^ { w }$ and the lowest-scoring candidate as $x _ { { \mathrm { ~ ~ i ~ } } } ^ { l }$ , giving a continually refreshed set $\mathcal { D } _ { \mathrm { o n } }$ that targets the current failure modes. The sampling policy $\pi _ { \mathrm { o l d } }$ is refreshed periodically from � so that generation tracks the improving model, while the DPO reference $\theta _ { \mathrm { r e f } }$ remains the supervised checkpoint so that the regularizer keeps anchoring the policy and prevents the online updates from drifting too far from a trusted initialization (Wallace et al., 2024).

Mixed training. We optimize the ofline and online objectives jointly, so that the curated corpus keeps supplying reliable, human-grounded preferences while the on-policy pairs continually adapt to the evolving policy:

$$
\mathcal { L } _ { \mathrm { D P O - m i x } } = \mathcal { L } _ { \mathrm { D P O } } ( \mathcal { D } _ { \mathrm { o f f } } ) + \lambda _ { \mathrm { o n } } \mathcal { L } _ { \mathrm { D P O } } ( \mathcal { D } _ { \mathrm { o n } } ) ,\tag{2}
$$

where $\lambda _ { \mathrm { o n } }$ balances the online term against the ofline term. We warm up on the ofline corpus alone and then anneal $\lambda _ { \mathrm { o n } }$ from 0 to 0.5 as on-policy pairs accumulate, so that the online signal takes over only once the policy is reliable enough to generate informative pairs.

Mixed DPO gives a stable, preference-aligned checkpoint, but it still reduces each group of candidates to a single best-versus-worst pair and supervises the model through a binary comparison, discarding both the graded magnitude of the reward and the information in the remaining candidates. We therefore hand of to online reinforcement learning with DifusionNFT, which consumes the full graded reward over every candidate in the group and optimizes it directly on the forward difusion process. In efect, DPO provides a robust warm start from paired preferences, and DifusionNFT extracts a finer, continuous learning signal from the same on-policy generations once the policy is strong enough to benefit from it.

## 3.4.2. DifusionNFT

We adopt DifusionNFT (Zheng et al., 2026) for online reinforcement learning from the DPO checkpoint. It optimizes the forward difusion process through flow matching (Lipman et al., 2023), using clean generated images and their rewards without estimating likelihoods or retaining denoising trajectories. For each editing condition $c ,$ the sampling policy generates a group of candidates whose rewards are normalized into optimality probabilities $r \in [ 0 , 1 ]$ . The optimization objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { N F T } } = \mathbb { E } _ { t , x _ { 0 } \sim \pi _ { \mathrm { o l d } } , \epsilon } \left[ r \left\| \boldsymbol { v } _ { \boldsymbol { \theta } } ^ { + } ( \boldsymbol { x } _ { t } , c , t ) - \boldsymbol { v } \right\| _ { 2 } ^ { 2 } + ( 1 - r ) \left\| \boldsymbol { v } _ { \boldsymbol { \theta } } ^ { - } ( \boldsymbol { x } _ { t } , c , t ) - \boldsymbol { v } \right\| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{3}
$$

where $x _ { t }$ is obtained by adding noise � to a generated sample $x _ { 0 } .$ , and � is the target velocity. The implicit policies satisfy $\nu _ { \theta } ^ { \pm } = \nu _ { \mathrm { o l d } } \pm \beta \left[ \nu _ { \theta } - \nu _ { \mathrm { o l d } } \right]$ , with all velocity fields evaluated at the same input and $\beta$ controlling guidance strength. The sampling policy $\nu _ { \mathrm { o l d } }$ is frozen during each update and refreshed by an exponential moving average of $\nu _ { \theta }$ . We use the VLM judge in §3.4.3 for general editing quality and combine it with the corresponding CTR, OCR, or consistency reward for each specialized objective.

## 3.4.3. VLM as Judge

We use Gemini 3.1 pro preview directly through its API as a fixed vision-language judge without additional fine-tuning. The judge receives the source image, any additional reference images, the editing instruction, and the generated result. A common rubric defines three integer scores from 1 to 5 for image quality, instruction alignment, and aesthetics. These dimensions assess rendering integrity, editing correctness, and visual presentation, respectively. The judge evaluates each dimension independently and provides a brief justification grounded in visible evidence.

Image quality. Rendering fidelity is evaluated locally within the edited region and globally across the image. The assessment considers clarity and structural integrity, including geometric distortions, texture discontinuities, and inconsistencies in lighting or occlusion. Scores reflect the severity and spatial extent of these defects, with higher values indicating cleaner rendering and more coherent integration of the edit. Judgments are made within the intended visual style.

Instruction alignment. To assess instruction alignment, the judge compares the generated result with the source and reference images under the specified editing instruction. The comparison checks the requested operations, target attributes, and spatial relations, together with the preservation of subject identity and content outside the intended edit. Preservation is judged relative to the intended transformation, allowing changes necessary to carry out the instruction. Missing requirements, incorrect targets, and unrelated alterations reduce the score, which jointly reflects editing correctness and completeness.

Aesthetics. The aesthetic assessment considers how composition, color, and tone jointly organize the visual presentation. Framing, spacing, and contrast are examined to determine whether they establish a clear focal subject and a balanced relationship with the background. In product images, this assessment places particular emphasis on product visibility and the arrangement of supporting elements. High scores require these elements to form a cohesive presentation, with harmonious color and tonal relationships, consistent styling, and minimal distraction from the intended subject.

## 3.4.4. CTR Reward Model

General editing quality does not capture whether a product image will attract clicks in a live storefront (Wang et al., 2021; Yang et al., 2024). Commercial appeal is a platform-specific signal that a general vision-language judge cannot score reliably, so we train a dedicated click-through rate (CTR) reward model that predicts the relative appeal of a candidate product image.

Data Collection and Filtering. We build the training set from platform behavior logs. For each product we gather the set of display images that have been served to users together with their logged impressions and clicks, and we form the empirical CTR of an image as its click count divided by its impression count. Two properties of raw logs make this signal noisy. An image with few impressions has a high-variance CTR estimate, and images of diferent products are not directly comparable because CTR is confounded by product category, price, and demand. We therefore filter the logs in two ways. First, we retain only images with more than 2,000 impressions to reduce noise in CTR estimates caused by low impression counts. Second, we build training examples only from pairs of images of the same product, which cancels product-level confounders and reduces the problem to ranking presentations of one item. We keep a pair only when its CTR ordering is consistent across all three observation windows of 7, 14, and 30 days, filtering out pairs whose ordering changes across these windows due to short-term trafic fluctuations.

Model Construction. The reward model encodes visual and textual product information at multiple granularities. On the visual side it encodes the full image together with a grid of local patches, which lets the model attend both to global composition and to local regions such as the product foreground and any rendered text. On the textual side it encodes the product title together with its multi-level category path. The visual and textual streams are encoded independently and fused by cross-attention, and a lightweight head maps the fused representation to a scalar score �. Encoding the two modalities separately before fusion avoids diluting the visual signal, which we find to be the primary carrier of commercial appeal.

Training Schemes. We train the reward model in two stages on the filtered same-product pairs. In the first stage we discretize observed CTR into � buckets and train a bucket classifier with crossentropy, which gives the encoder a coarse but stable notion of click attractiveness before it is exposed to fine-grained ranking. In the second stage we refine the model with a gap-weighted pairwise ranking loss combined with a pointwise regression term:

$$
\mathcal { L } _ { \mathrm { C T R } } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , j ) \in \mathcal { P } } w _ { i j } \log \left( 1 + e ^ { - ( s _ { i } - s _ { j } ) } \right) + \lambda \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( s _ { i } - \hat { c } _ { i } \right) ^ { 2 } , \qquad w _ { i j } = \operatorname* { m i n } \biggl ( 1 , \frac { | c _ { i } - c _ { j } | } { \gamma } \biggr ) ,\tag{4}
$$

where $s _ { i }$ is the predicted score, $c _ { i }$ the observed CTR, $\hat { c } _ { i }$ the regression target, $\mathcal { P }$ the set of same-product pairs with $c _ { i } > c _ { j } , \gamma$ a gap normalization constant, and � the regression weight. The pairwise term follows the logistic ranking formulation of Burges et al. (2005) and teaches the model the ordering within each product, and the pointwise regression term keeps the absolute scores calibrated across products. The gap-aware weight $w _ { i j }$ down-weights near-tie pairs, whose ordering is dominated by measurement noise, and emphasizes pairs with a clear CTR diference, which acts as an implicit curriculum. We set $K = 1 0$ buckets and $\lambda = 0 . 2$

Table 3 | Pairwise ranking accuracy on the public CreativeRanking benchmark. Our vision-only variant (vision encoder plus MLP head, with the text branch and fusion module removed) is compared against general-purpose and task-specific baselines.
<table><tr><td>Model</td><td>PairAcc</td></tr><tr><td>Qwen3-VL</td><td>47.9</td></tr><tr><td>GPT-40</td><td>50.2</td></tr><tr><td>Qwen3-VL-Trained</td><td>51.2</td></tr><tr><td>VAM</td><td>52.2</td></tr><tr><td>LLaVA</td><td>52.8</td></tr><tr><td>CG4CTR</td><td>53.1</td></tr><tr><td>CAIG</td><td>56.2</td></tr><tr><td>Ours (Vision-only)</td><td>57.1</td></tr></table>

Method Comparison. We first compare the learned reward model against using a general-purpose vision-language model as a zero-shot CTR judge. General judges perform close to chance on pairwise CTR ranking, because commercial appeal is a platform-specific signal that is not recoverable from generic visual-text pretraining, whereas the dedicated model attains roughly 70% pairwise ranking accuracy on our held-out pairs. To position the model against prior work under a common protocol, we further evaluate on the public CreativeRanking benchmark (Wang et al., 2021), a large-scale creative-ranking dataset of over 1.7M ad creatives from 500K products. Because this benchmark provides only image input, we reduce our model to a vision-only variant. Even in this reduced form the model reaches 57.1 pairwise accuracy, surpassing both general-purpose baselines (Qwen3-VL 47.9, GPT-4o 50.2, LLaVA 52.8) and task-specific creative-ranking methods (VAM (Wang et al., 2021) 52.2, CG4CTR (Yang et al., 2024) 53.1, CAIG (Chen et al., 2025) 56.2), as summarized in table 3. This confirms that the training strategy, which first learns CTR buckets and then refines pairwise rankings, is the primary source of the model’s ranking ability, rather than multimodal fusion alone.

The model also holds a clear eficiency advantage. It scores an image with a single forward pass through a compact SigLIP2-base encoder (Tschannen et al., 2025) and a lightweight head, which is orders of magnitude cheaper than prompting a multi-billion-parameter vision-language judge once per candidate, and this cost diference is decisive when the model is used both as an online training reward over large candidate groups and as an ofline filter across the full production catalog. Scaling the encoder further brings little benefit: replacing SigLIP2-base with SigLIP2-giant improves pairwise accuracy by only about 0.2 points, which indicates that the accuracy comes from the architecture and the training strategy rather than from encoder capacity, so a small, fast encoder is suficient in practice. Ablations further show that the visual encoder is the primary information bottleneck, that overly fine bucket discretization introduces label noise, and that the soft gap-aware weight outperforms hard filtering of near-tie pairs. Taken together, these results justify a purpose-built, compact reward model over a prompted general judge for the commercial objective.

Efect on Generated Image Usability. We use the CTR reward model to score and select product images. In an online A/B experiment, this selection yields an approximately 2.44% relative increase in observed CTR over the control group. We evaluate how reinforcement learning with the CTR reward changes the usability of generated product images on a fixed test set. For each source image and editing instruction, the editing models before and after CTR reward optimization each produce a candidate under the same generation protocol. The original product image serves as the common control, and each model’s output is separately paired with this control. We define CTR-based usability as the fraction of evaluated pairs in which the generated image receives a strictly higher CTR than its original counterpart. Ties do not count as improvements, and both checkpoints are evaluated over the same test cases. Under this criterion, the usable proportion increases from 12.16% before CTR reward optimization to 37.41% afterward, an absolute gain of 25.25 percentage points. Both rates measure improvement over the original product images, rather than direct wins between the two editing models. This result shows that CTR reward optimization increases the fraction of generated candidates preferred to the original images. This is an ofline measure of generated image usability. Together, the ofline generation evaluation and the online selection experiment support the CTR model’s utility in guiding image generation and identifying visual assets with greater click potential.

## 3.4.5. OCR Reward Model

Product images frequently carry dense text such as prices, promotions, and selling points, and errors in rendered text are immediately visible to shoppers (Tuo et al., 2024). String-level recognition rewards are insensitive to glyph-level defects such as missing strokes or distorted characters that harm legibility without changing the recognized string. We therefore adopt a coarse-to-fine text reward that combines a span-level term for semantic placement with a glyph-level term for structural fidelity (Wang et al., 2026a).

The span-level reward evaluates whether the intended text spans appear at scene-appropriate locations without spurious or missing content. Detected text is first filtered to remove detections that overlap existing source text or the foreground product region, and each target span is matched to a detection through a normalized edit distance similarity. The span reward is the product of a fidelity term over matched spans and a coverage term that penalizes both missing target spans and spurious detections:

$$
R _ { \mathrm { s p a n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r _ { i } \cdot \mathrm { m a x } \Bigg ( 0 , 1 - \frac { U _ { \mathrm { t a r } } + U _ { \mathrm { o c r } } } { \mathrm { m a x } ( N , | V | ) } \Bigg ) .\tag{5}
$$

where $r _ { i }$ is the per-span similarity of the �-th of � target spans, $U _ { \mathrm { t a r } }$ and $U _ { \mathrm { o c r } }$ are the numbers of unmatched target and detected spans, and � is the set of valid detections. To mitigate reward hacking under prolonged optimization and preserve the appearance of the rest of the image, we add a gated structural regularizer that activates only once span accuracy is suficiently high:

$$
R _ { \mathrm { t e x t } } = R _ { \mathrm { s p a n } } + \mathbb { 1 } \left[ R _ { \mathrm { s p a n } } > \tau _ { \mathrm { s s i m } } \right] \cdot S S \mathrm { I M } \big ( \psi ( \hat { x } , V ) , \psi ( x _ { \mathrm { s r c } } , V ) \big ) ,\tag{6}
$$

where $\psi ( \cdot , V )$ fills the retained text boxes in � with white before the comparison, so that the SSIM term (Wang et al., 2004) measures structural preservation of the non-text regions alone rather than of the edited text itself, and $\tau _ { s s \mathrm { i m } }$ gates the regularizer so that span correctness remains the primary objective.

The glyph-level reward provides dense supervision on character structure. For a target glyph with an annotated character box, the region is cropped and passed to an OCR recognizer, and the reward is the maximum posterior assigned to the target glyph across the recognizer’s output timesteps:

$$
R _ { \mathrm { g l y p h } } = \operatorname* { m a x } _ { 1 \leq t \leq T } P _ { t , { \mathrm { i d } } ( y ^ { \star } ) } ,\tag{7}
$$

where $P \in [ 0 , 1 ] ^ { T \times | C | }$ is the recognizer’s per-timestep character posterior matrix and id $( y ^ { \star } )$ is the vocabulary index of the target glyph. This graded signal decreases smoothly under stroke omission or structural distortion, which a binary recognition reward cannot express. The two rewards are applied by task type on a mixed stream, with the span-level reward supervising text insertion and the glyph-level reward supervising text replacement, and both are converted to the optimality probability consumed by eq. (3).

## 3.4.6. Fine-grained Consistency Reward Model

Many e-commerce edits must preserve the identity of a product or a model across a substantial change of background, pose, or viewpoint, and pixel reconstruction losses do not capture failures such as color drift, implausible textures, or subtle identity deviation. A reward that only checks whether the overall subject looks like the same product is also insuficient, because the defects that matter commercially are often local: a mismatched cuf, a wrong button, a fabric weave that does not correspond to the source, or a detail region that is rendered plausibly but is not the part that was requested (Li et al., 2026b). We therefore design a consistency reward that operates at two levels of granularity, and this multi-level formulation is a central contribution of our reward design. Concretely the reward scores a reference and a generated image along two complementary axes and combines them into a single scalar:

$$
r = \alpha _ { \mathrm { i d } } r _ { \mathrm { i d } } + \alpha _ { \mathrm { t f } } r _ { \mathrm { t f } } ,\tag{8}
$$

where the two axes capture, respectively:

• Product identity consistency $( r _ { \mathrm { i d } } )$ : whether the result depicts the exact same product as the reference, verified against unique, non-generic features such as the specific fabric material, stitching and seam structure, and hardware or trims, rather than mere category or color similarity.

• Target-part and detail fidelity $\left( r _ { \mathrm { t f } } \right)$ : whether the intended local part is depicted accurately and completely, with its local structure, spatial relationships, and fine process and decorative details preserved from the reference.

The reward thus verifies consistency not only at the level of the overall product subject, but also at the level of the requested part and of the fine-grained details within it. This part- and detail-level supervision is what distinguishes our reward from generic identity-oriented preference models, and it directly targets the local mismatches that determine whether a commercial detail image is usable.

Reward model and training data. The reward model is a vision-language model fine-tuned to emit the two integer scores under a fixed, behavior-anchored rubric, taking the reference image, the candidate image, and the prompt as input. Each of the two axes is defined by an explicit 1-to-4 rubric with positive and negative indicators, and the rubric enforces a strict-uncertainty principle: generic similarity alone cannot earn a high score, and any uncertainty about feature matching or part accuracy must be resolved toward the lower score. This makes the learned reward conservative, which is desirable for an RL signal that would otherwise be easy to hack with superficially plausible but inconsistent edits.

We build the training corpus by re-annotating a large pool of reference-candidate pairs, covering both high-quality detail shots and deliberately imperfect generations, so that the model sees the full range of consistency levels rather than only near-perfect examples. A stronger vision-language model scores every pair on the two axes under the same rubric, and we retain only the integer scores as supervision. Two filtering steps keep the labels reliable: pairs on which the annotator’s rationale contradicts its score are discarded, and a held-out subset is checked so that the model separates known-consistent from known-inconsistent pairs on the identity and part-fidelity axes before it is used as a reward. We fine-tune the model for a small number of epochs on this filtered corpus.

During reinforcement learning the composite reward � is normalized within each candidate group into the optimality probability consumed by eq. (3), and the reference term retained from the earlier alignment stages keeps the update from eroding the model’s editing ability.

## 3.4.7. DifusionOPD

The capability-specific objectives described above are optimized through separate reinforcement learning runs. Each objective is defined by a composite reward that combines the general-purpose VLM judge with one specialized reward model. These objectives pair the VLM judge with the corresponding specialized reward models. The VLM judge provides a shared assessment of general output quality and reduces the risk of overoptimizing exploitable patterns in any single specialized reward. Separate optimization is preferable because jointly training on heterogeneous objectives can lead to cross-task interference and imbalance in optimization dificulty. Sequentially optimizing these objectives with a single policy can instead cause catastrophic forgetting. We therefore obtain one specialized teacher for each composite task objective and consolidate the teachers into a unified model using DifusionOPD (Li et al., 2026a), an on-policy distillation method for multi-task difusion training.

DifusionOPD separates task-specific exploration from multi-task integration. In the first stage, we train task-specific teachers using DifusionNFT. Each teacher is optimized for one composite objective and jointly considers the VLM judge and the corresponding specialized reward model. This design preserves the specialization induced by the task-specific reward while maintaining a shared notion of general output quality across all teachers. In the second stage, we distill the teachers into a single student along trajectories generated by the student itself.

Let � index the task-specific teachers. Since the student and each teacher use the same denoising schedule, their one-step transition kernels are Gaussian distributions with a shared covariance and difer only in their means. The per-step reverse Kullback–Leibler divergence therefore reduces to the following mean-matching objective.

$$
\mathcal { L } _ { \mathrm { O P D } } = \mathbb { E } _ { k \sim p ( k ) } \mathbb { E } _ { \boldsymbol { x } _ { 0 : N } \sim p _ { S , \theta } ^ { k } } \left[ \sum _ { j = 0 } ^ { N - 1 } \frac { \left. \boldsymbol { \mu } _ { S } ^ { k } ( \boldsymbol { x } _ { t _ { j } } ; \theta ) - \boldsymbol { \mu } _ { T _ { k } } ( \boldsymbol { x } _ { t _ { j } } ) \right. _ { 2 } ^ { 2 } } { 2 \bar { \sigma } _ { j } ^ { 2 } } \right] ,\tag{9}
$$

where $\mu _ { s } ^ { k }$ denotes the student transition mean for task �, $\mu _ { T _ { k } }$ denotes the transition mean of the corresponding teacher, and $\bar { \sigma } _ { j } ^ { 2 }$ is the shared variance at denoising step �. The trajectories are sampled on-policy from the student under the data distribution of each task. Each teacher therefore supervises the states that the student actually visits for the corresponding task. During each training round, we collect the distillation loss for every task and apply one update using the aggregated loss. This balanced update prevents the consolidated student from being dominated by any single task objective.

## 4. Benchmark

To address the requirements of both practical business scenarios (Fan et al., 2026; Li et al., 2026b; Wang et al., 2026a) and general-purpose image editing (Wei et al., 2025; Xia et al., 2025a; Yu et al., 2025), we need a reliable evaluation protocol that characterizes model behavior across domainspecific commercial applications and broad editing tasks. We therefore evaluate KwaiMind in two domains: (1) the general domain, using established image editing benchmarks that measure broad instruction-following ability, and (2) the e-commerce domain, using our 11-task Ecom-Bench with domain-tailored evaluation criteria.

## 4.1. General Image Editing Benchmark

We assess general editing competence on three complementary public benchmarks: ImgEdit-Bench Ye et al. (2025), which evaluates instruction following and visual quality with task-adaptive scores;

GEdit-Bench Liu et al. (2025), which combines semantic consistency and perceptual quality using VIEScore Ku et al. (2024); and REDEdit-Bench Team et al. (2026), which assesses diverse editing tasks with parallel Chinese and English instructions, evaluated separately. ImgEdit averages dimension scores per sample and then across editing categories, while GEdit uses the geometric mean of semantic consistency and perceptual quality. Because the original ImgEdit and GEdit judges produced unstable scores across repeated evaluations, we use Gemini 3.1 Pro Preview for both, keeping their evaluation prompts and scoring rules unchanged.

## 4.2. Ecom-Bench: E-commerce Image Editing Benchmark

General-purpose editing benchmarks do not reflect the distinctive requirements of commercial image production. Tasks such as virtual try-on, selling-point poster composition, garment texture replication, and tagline removal must jointly meet requirements for product identity fidelity, typographic accuracy, and commercial visual appeal. General benchmarks neither isolate nor measure these criteria. We therefore introduce Ecom-Bench, a benchmark purpose-built for the e-commerce domain, covering 11 editing tasks under a systematic and domain-tailored evaluation protocol.

## 4.2.1. Task Taxonomy

Ecom-Bench organizes e-commerce image editing into three thematic groups spanning 11 tasks in total:

• Garment & Wearable (5 tasks): Virtual Try-On, Clothing Detail, Clothing Display, Universal Wearing, Pose Change.

• Composition & Layout (3 tasks): Background Replace, Outpaint, Product Extract.

• Text Operations (3 tasks): Text Edit, Tagline Removal, Selling Point Display.

Table 4 summarizes the definition and key properties of each task. The multi-reference tasks (Virtual Try-On and Universal Wearing) take two images as input — a model image and a product image — matching the paired-reference input format used during training.

Table 4 | The 11 tasks of Ecom-Bench. <sup>†</sup> denotes tasks with multi-image reference input (model image + product image). 100 test samples are used per task.

<table><tr><td>Task</td><td>Description</td></tr><tr><td>Virtual Try-On† Clothing Detail Clothing Display Universal Wearing†</td><td>Dress a model image with a specified garment Generate a zoomed-in detail view of garment texture Clothe a virtual mannequin model with a target garment Apply any wearable product to a model image</td></tr><tr><td>Pose Change Background Replace Outpaint Product Extract</td><td>Alter the model&#x27;s pose while preserving garment appearance Swap the product backdrop with a specified scene Coherently extend the canvas of a product display image Segment and extract the product with clean edges</td></tr><tr><td>Text Edit Tagline Removal Selling Point Display</td><td>Modify or add text overlays on a product display image Remove promotional taglines from product images</td></tr></table>

## 4.2.2. Benchmark Curation

Ecom-Bench is curated from diverse real and synthetic e-commerce imagery to cover varied products, presentation styles, and editing conditions. Candidate images are filtered for visual quality and task suitability, after which task-specific templates and VLM assistance are used to construct contextappropriate editing instructions. For the multi-reference tasks, Virtual Try-On and Universal Wearing, the benchmark pairs a model image with a target product image. All resulting image–instruction samples undergo a final quality audit before inclusion.

Each sample follows the format {edit\_image, prompt, info}, where edit\_image contains either one image or a list of images for multi-reference tasks. This consistent representation allows all 11 tasks to share the same inference and evaluation pipeline while retaining their task-specific input requirements.

## 4.2.3. Evaluation Metrics

Evaluation Dimension Library. We define a library of 14 evaluation dimensions, consisting of 6 general dimensions applicable across all editing contexts and 8 e-commerce-specific dimensions that target domain-critical quality attributes.

## General dimensions (G):

• G1 Instruction Compliance — fidelity of the output to the editing instruction.

• G2 Visual Naturalness & Seamlessness — naturalness of compositing boundaries and absence of visible blending seams.

• G3 Physical & Lighting Plausibility — geometric and photometric coherence of shadows, reflections, and perspective.

• G4 Non-edited Region Preservation — pixel-level retention of regions outside the intended edit.

• G5 Image Quality — sharpness and freedom from noise and visual artifacts.

• G6 Overall Aesthetics — conformance to commercial presentation standards.

## E-commerce-specific dimensions (E):

• E1 Product Identity Consistency — preservation of the product’s appearance and brand identity.

• E2 Text Accuracy — legibility, spelling correctness, and positioning of text elements.

• E3 Texture & Fabric Fidelity — accuracy of reproduced material textures and patterns.

• E4 Model Identity & Pose Naturalness — naturalness of the model’s appearance and body posture.

• E5 Garment Fit & Silhouette — realism of garment drape and fit on the body.

• E6 Cutout Precision & Edge Quality — cleanliness and precision of segmentation boundaries.

• E7 Key Selling Point Conveyance — clarity with which the highlighted product features are emphasized.

• E8 Product Presentation Completeness — display of the complete product, free of unintended cropping or omission.

Per-task Dimension Assignment. Each task is assigned exactly four dimensions under a 2-general + 2-e-commerce-specific configuration. Table 5 lists the dimension assignments for all 11 tasks.

Table 5 | Evaluation dimension assignments for the 11 Ecom-Bench tasks. The G and E prefixes denote general and e-commerce-specific dimensions, respectively.
<table><tr><td>Task</td><td>D1</td><td>D2</td><td>D3</td><td>D4</td></tr><tr><td>Virtual Try-On</td><td>E5 Fit &amp; Silhouette</td><td>E3 Texture</td><td>G2 Seamless</td><td>G3 Physical</td></tr><tr><td>Clothing Detail</td><td>E1 Product Id.</td><td>E3 Texture</td><td>G5 Quality</td><td>G6 Aesthetics</td></tr><tr><td>Clothing Display</td><td>E3 Texture</td><td>E4 Model</td><td>G3 Physical</td><td>G6 Aesthetics</td></tr><tr><td>Universal Wearing</td><td>E1 Product Id.</td><td>E4 Model</td><td>G2 Seamless</td><td>G3 Physical</td></tr><tr><td>Pose Change</td><td>E4 Model</td><td>E5 Fit &amp; Silhouette</td><td>G1 Comply</td><td>G3 Physical</td></tr><tr><td>Background Replace</td><td>G1 Comply</td><td>E1 Product Id.</td><td>E8 Complete</td><td>G3 Physical</td></tr><tr><td>Outpaint</td><td>G4 Preserve</td><td>E8 Complete</td><td>E1 Product Id.</td><td>G2 Seamless</td></tr><tr><td>Product Extract</td><td>E6 Cutout</td><td>E1 Product Id.</td><td>G1 Comply</td><td>G5 Quality</td></tr><tr><td>Text Edit</td><td>E2 TextAcc.</td><td>E1 Product Id.</td><td>G2 Seamless</td><td>G4 Preserve</td></tr><tr><td>Tagline Removal</td><td>G1 Comply</td><td>E1 Product Id.</td><td>E8 Complete</td><td>G2 Seamless</td></tr><tr><td>Selling Point Display</td><td>E7 SellingPt.</td><td>E2 TextAcc.</td><td>G1 Comply</td><td>G6 Aesthetics</td></tr></table>

Scoring and Aggregation. Each dimension is rated on a 1–5 integer scale. Every score level is specified by a concrete behavioral rubric that describes what an output at that level looks like, rather than relying on abstract adjectives such as “good” or “poor”; this behavior-anchored design reduces inter-run variance and score drift under a stochastic VLM judge. Evaluation is performed by Gemini 3.1 Pro Preview, which is presented with the reference image(s), the editing instruction, and the generated output.

The per-sample composite score is the geometric mean of the four dimension scores:

$$
S = \left( d _ { 1 } \cdot d _ { 2 } \cdot d _ { 3 } \cdot d _ { 4 } \right) ^ { 1 / 4 } , \quad d _ { i } \in \{ 1 , 2 , 3 , 4 , 5 \} .\tag{10}
$$

We deliberately adopt the geometric mean over the arithmetic mean, as it penalizes dimension-level failures more severely. This behavior is consistent with commercial quality requirements, under which a single critical defect — garbled product text, loss of product identity, or an unnaturally fitted garment — renders an image unusable regardless of its scores on the other dimensions. Task-level scores are obtained by averaging � over the samples of each task, and the overall Ecom-Bench score is the macro-average over the 11 tasks.

CTR Score. We use the CTR reward model described in §3.4.4 to evaluate the predicted commercial attractiveness of generated images across all 11 Ecom-Bench tasks. Trained on product impression and click logs through CTR bucket classification followed by pairwise ranking refinement, the model provides a complementary measure of learned click preferences.

For comparison, we retain only test cases completed by all models included in the CTR evaluation and rank their outputs by the reward model’s scores within each case. For each model, we report the total number of appearances at ranks 1, 2, and 3, with equal weight assigned to each rank. This statistic measures how frequently a model’s output ranks among the top three under the learned CTR reward model; it does not represent measured CTR or actual click improvement.

## 4.2.4. Evaluation Protocol

We use Gemini 3.1 Pro Preview as the automated evaluator under a task-specific prompting protocol. For each sample, the judge is presented with the reference image(s), the editing instruction, and the generated output, and assigns scores for the four dimensions selected for that task. Virtual Try-On and

Table 6 | Results on ImgEdit and GEdit under a common Gemini 3.1 Pro Preview judging protocol. All task columns and the overall score are reported. Within the open-source and closed-source groups separately, the best result is shown in bold and the second-best is underlined.  
Panel A: ImgEdit (Gemini judge)
<table><tr><td>Model</td><td>Add</td><td>Adjust</td><td>Remove</td><td>Replace</td><td>Back.</td><td>Style</td><td>Extract</td><td>Action</td><td></td><td>Compose</td><td>Overall</td></tr><tr><td>Flux-2.0</td><td>4.21</td><td>3.61</td><td>4.07</td><td>4.36</td><td></td><td></td><td></td><td>3.13</td><td>3.23</td><td>2.26</td><td>3.69</td></tr><tr><td>Joy-Edit</td><td>4.34</td><td>4.06</td><td>4.25</td><td>4.25</td><td>3.78 4.00</td><td></td><td></td><td>4.40 2.22</td><td>3.10</td><td>3.44</td><td>4.09</td></tr><tr><td>Joy-Edit-Plus</td><td>4.41</td><td>4.24</td><td>3.62</td><td>3.67</td><td>3.53</td><td></td><td></td><td></td><td>3.39</td><td>2.60</td><td>3.57</td></tr><tr><td>LongCat</td><td>4.28</td><td>4.07</td><td>4.29</td><td>4.44</td><td>4.05</td><td>4.41 4.98</td><td></td><td></td><td>3.00</td><td>3.47</td><td>4.10</td></tr><tr><td>FireRed-Edit</td><td>4.36</td><td>4.11</td><td>4.40</td><td>4.59</td><td>3.94</td><td>4.96</td><td>4.38 4.84 4.27</td><td></td><td>2.89</td><td>2.90</td><td>4.11</td></tr><tr><td>Qwen-Edit-2511</td><td>4.44</td><td>3.63</td><td>3.86</td><td>4.20</td><td>3.47</td><td>4.82 4.89</td><td></td><td></td><td>2.73</td><td>3.01</td><td>3.83</td></tr><tr><td>KwaiMind</td><td>4.33</td><td>3.87</td><td>4.26</td><td>4.47</td><td>4.24</td><td></td><td></td><td>4.50</td><td>3.27</td><td>3.52</td><td>4.15</td></tr><tr><td>Seedream</td><td>4.53</td><td>4.26</td><td>4.27</td><td>4.52</td><td></td><td></td><td>4.96</td><td>3.24</td><td>4.49</td><td>3.87</td><td>4.27</td></tr><tr><td>NanoBanana-2</td><td>4.54</td><td>4.52</td><td>4.20</td><td>4.50</td><td>4.29 4.06</td><td>4.98 5.00</td><td></td><td></td><td>2.85</td><td>3.38</td><td>4.09</td></tr><tr><td>GPT-Image-2</td><td>4.68</td><td colspan="3">4.60 4.34</td><td>4.52</td><td></td><td></td><td colspan="2">3.76 4.37 3.17</td><td>2.41</td><td>4.20</td></tr><tr><td colspan="10">Panel B: GEdit (Gemini judge)</td><td></td><td></td><td></td></tr><tr><td>Model</td><td>Back.</td><td>Color</td><td>Material</td><td>Motion</td><td>Human</td><td>Style</td><td>Add</td><td>Remove Replace</td><td>Text</td><td>Tone</td><td>Overall</td></tr><tr><td>Flux-2.0</td><td>5.07</td><td>6.01</td><td>5.71</td><td>5.02</td><td>4.68</td><td>5.94</td><td></td><td></td><td></td><td>6.38</td><td>5.34</td></tr><tr><td>Joy-Edit</td><td>5.46</td><td>5.13</td><td>5.15</td><td>4.39</td><td>3.97 5.70</td><td>5.44 5.62</td><td>4.53 5.91</td><td>4.73 5.42</td><td>5.23 6.26</td><td>5.47</td><td>5.32</td></tr><tr><td>Joy-Edit-Plus</td><td>3.25</td><td>4.29</td><td>3.65</td><td>3.72</td><td>3.38 4.12 5.60</td><td>4.58</td><td>2.73</td><td>3.14</td><td>3.95</td><td>4.56</td><td>3.76</td></tr><tr><td>LongCat</td><td>5.24</td><td>4.99</td><td>4.94</td><td>4.00</td><td>3.87</td><td>5.54 5.83</td><td>6.09</td><td>4.95</td><td>6.17</td><td>5.35</td><td>5.16</td></tr><tr><td>FireRed-Edit Qwen-Edit-2511</td><td>5.91</td><td>6.11</td><td>6.08</td><td>5.69</td><td>5.92</td><td>6.17 6.20</td><td>6.07 6.00</td><td>5.35</td><td>5.85</td><td>5.87 5.82</td><td>5.90 5.97</td></tr><tr><td>KwaiMind</td><td>5.75 5.55</td><td>6.15</td><td>6.15</td><td>6.18</td><td>5.83</td><td>6.24 6.41</td><td>6.20</td><td>5.64 6.27</td><td>5.69 6.36</td><td>5.85</td><td>6.29</td></tr><tr><td></td><td></td><td>6.37</td><td>6.42</td><td>6.83</td><td>6.46</td><td>6.49</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Seedream</td><td>5.99</td><td>6.11</td><td>6.20</td><td>6.24</td><td>5.54</td><td>6.46 6.97</td><td>5.95</td><td>6.21</td><td>6.47</td><td>5.89</td><td>6.10</td></tr><tr><td>NanoBanana-2</td><td>6.38</td><td>7.06</td><td>6.92</td><td>7.19</td><td>7.24</td><td>6.08 6.74</td><td>6.85</td><td>7.21</td><td>7.47</td><td>7.19</td><td>7.02</td></tr><tr><td>GPT-Image-2</td><td>6.83</td><td>7.19</td><td>6.78</td><td>7.29</td><td>7.56</td><td>6.96</td><td>6.92</td><td>7.39</td><td>7.25</td><td>6.75</td><td>7.10</td></tr></table>

Universal Wearing provide both the model and product images as references. The dimension scores are combined and aggregated as described above to obtain the per-task and overall Ecom-Bench results.

## 5. Evaluation

We compare KwaiMind with six open-source image editors: Flux.2-dev Black Forest Labs (2025), Joy-Image-Edit, Joy-Image-Edit-Plus Song et al. (2026), LongCat Meituan LongCat Team et al. (2025), FireRed-Image-Edit-1.0, and Qwen-Image-Edit-2511 Wu et al. (2025a), and three additional closedsource systems: Seedream5.0, NanoBanana-2, and GPT-Image-2. To make the comparison internally consistent, ImgEdit and GEdit are evaluated with the same Gemini 3.1 Pro Preview judge for every model, while REDEdit follows its original English and Chinese evaluation protocol. Ecom-Bench reports both the task-specific visual score defined in §4.2.3 and a CTR-based comparison over the test cases completed by all models included in the CTR evaluation. For each test case, the model outputs are ranked by their predicted CTR scores; appearances at rank 1, rank 2, and rank 3 are counted, and their equal-weight sum is reported for each model. In all tables, open-source and closed-source models are separated by a horizontal rule and ranked independently: the best score within each group is bolded and the second-best is underlined.

## 5.1. General Image Editing Results

Figure 3 gives an overview of the aggregate results, while Tables 6 and 7 provide the complete tasklevel breakdown. KwaiMind achieves the best overall result among open-source models on all four general benchmarks: 4.15 on ImgEdit, 6.29 on GEdit, 3.75 on REDEdit English, and 3.73 on REDEdit Chinese. The improvements are not confined to a single edit family. On ImgEdit, KwaiMind leads the open-source group on background and compositional editing and ranks second on replacement and extraction. On GEdit, it obtains the strongest open-source result on material alteration, motion change, human editing, style change, subject removal, and subject replacement. This breadth indicates that continued pre-training preserves general editing competence despite the subsequent specialization toward e-commerce data.

Table 7 | Results on the English and Chinese splits of REDEdit. Within the open-source and closedsource groups separately, the best result is shown in bold and the second-best is underlined.
<table><tr><td>Model</td><td>Add</td><td>Adjust</td><td>Back.</td><td>Beauty</td><td>Color</td><td>Compose</td><td>Extract</td><td>Portrait</td><td>Low</td><td>Motion</td><td>Remove</td><td>Replace</td><td>Style</td><td>Text</td><td>View</td><td>Overall</td></tr><tr><td>Flux-2.0</td><td>3.96</td><td>3.32</td><td>4.09</td><td>3.02</td><td>3.28</td><td>2.96</td><td>1.31</td><td>3.79</td><td>3.66</td><td>4.30</td><td>3.43</td><td>4.12</td><td>4.41</td><td>2.75</td><td>2.95</td><td>3.42</td></tr><tr><td>Joy-Edit</td><td>3.95</td><td>3.32</td><td>3.90</td><td>2.29</td><td>3.58</td><td>3.01</td><td>2.46</td><td>3.54</td><td>3.11</td><td>3.86</td><td>3.64</td><td>4.10</td><td>4.75</td><td>3.28</td><td>1.67</td><td>3.36</td></tr><tr><td>Joy-Edit-Plus</td><td>2.90</td><td>1.69</td><td>2.43</td><td>1.42</td><td>1.93</td><td>2.22</td><td>1.07</td><td>3.42</td><td>2.20</td><td>3.38</td><td>1.83</td><td>2.10</td><td>3.18</td><td>2.25</td><td>1.75</td><td>2.25</td></tr><tr><td>LongCat</td><td>4.02</td><td>3.25</td><td>3.91</td><td>2.31</td><td>3.55</td><td>2.97</td><td>2.32</td><td>3.49</td><td>2.98</td><td>3.91</td><td>3.62</td><td>4.20</td><td>4.69</td><td>3.48</td><td>1.69</td><td>3.36</td></tr><tr><td>FireRed-Edit</td><td>4.37 4.20</td><td>3.75 3.21</td><td>4.27</td><td>2.82</td><td>3.92</td><td>3.51</td><td>2.38</td><td>3.60</td><td>2.89</td><td>4.33</td><td>4.10</td><td>4.33</td><td>4.77</td><td>3.66</td><td>2.39</td><td>3.67 3.43</td></tr><tr><td>Qwen-Edit-2511 KwaiMind</td><td>4.27</td><td></td><td>3.69</td><td>2.76</td><td>3.19</td><td>3.22</td><td>2.26</td><td>3.77</td><td>2.52</td><td>4.61</td><td>3.63</td><td>4.10</td><td>4.57</td><td>3.22</td><td>2.57</td><td>3.75</td></tr><tr><td></td><td></td><td>3.41</td><td>4.10</td><td>2.99</td><td>3.85</td><td>3.61</td><td>3.13</td><td>3.99</td><td>3.04</td><td>4.56</td><td>4.17</td><td>4.27</td><td>4.68</td><td>3.45</td><td>2.69</td><td></td></tr><tr><td>Seedream</td><td>4.49</td><td>3.86</td><td>4.35</td><td>3.48</td><td>4.29</td><td>3.84</td><td>2.12</td><td>4.47</td><td>3.80</td><td>4.63</td><td>4.23</td><td>4.57</td><td>4.91</td><td>4.30</td><td>2.55</td><td>3.99</td></tr><tr><td>NanoBanana-2</td><td>4.56</td><td>3.91</td><td>4.11</td><td>4.07</td><td>4.24</td><td>3.78</td><td>2.79</td><td>4.60</td><td>4.24</td><td>4.72</td><td>4.11</td><td>4.44</td><td>4.76</td><td>4.40</td><td>3.11</td><td>4.12</td></tr><tr><td>GPT-Image-2</td><td>4.73</td><td>4.03</td><td>4.45</td><td>4.01</td><td>4.11</td><td>3.97</td><td>3.40</td><td>4.43</td><td>3.40</td><td>4.83</td><td>4.19</td><td>4.57</td><td>4.98</td><td>4.40</td><td>3.00</td><td>4.17</td></tr><tr><td colspan="10">Panel B: REDEdit Chinese</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Model</td><td>Add</td><td>Adjust</td><td>Back.</td><td>Beauty</td><td>Color</td><td>Compose</td><td>Extract</td><td>Portrait</td><td>Low</td><td>Motion</td><td>Remove</td><td>Replace</td><td>Style</td><td>Text</td><td>View</td><td>Overall</td></tr><tr><td>Flux-2.0</td><td>3.85</td><td>3.19</td><td>4.09</td><td>2.79</td><td>3.25</td><td>2.92</td><td>1.26</td><td>4.06</td><td>3.66</td><td>4.15</td><td>3.06</td><td>3.88</td><td>4.38</td><td>2.64</td><td>2.44</td><td>3.31</td></tr><tr><td>Joy-Edit</td><td>3.76</td><td>3.08</td><td>3.75</td><td>2.05</td><td>3.36</td><td>2.91</td><td>1.81</td><td>3.38</td><td>2.69</td><td>3.73</td><td>3.33</td><td>3.92</td><td>4.67</td><td>3.27</td><td>1.88</td><td>3.17</td></tr><tr><td>Joy-Edit-Plus LongCat</td><td>2.86 3.72</td><td>1.78</td><td>2.37</td><td>1.38</td><td>1.81</td><td>1.98</td><td>1.09</td><td>3.18</td><td>1.98</td><td>3.12</td><td>1.46</td><td>1.84</td><td>3.03</td><td>2.10</td><td>1.90</td><td>2.13 3.17</td></tr><tr><td>FireRed-Edit</td><td>4.33</td><td>3.09 3.62</td><td>3.71 4.17</td><td>2.01 2.62</td><td>3.45 4.05</td><td>2.78 3.56</td><td>1.73 2.43</td><td>3.46 3.64</td><td>2.71 2.96</td><td>3.79 4.46</td><td>3.31 4.14</td><td>3.88 4.35</td><td>4.71 4.76</td><td>3.39 3.71</td><td>1.74 2.31</td><td>3.67</td></tr><tr><td>Qwen-Edit-2511</td><td>4.15</td><td>3.22</td><td>3.63</td><td>2.68</td><td>3.35</td><td>3.33</td><td>2.32</td><td>3.43</td><td>2.54</td><td>4.47</td><td>3.75</td><td>4.15</td><td>4.61</td><td>3.40</td><td>2.38</td><td>3.43</td></tr><tr><td>KwaiMind</td><td>4.26</td><td>3.41</td><td>4.13</td><td>2.84</td><td>3.81</td><td>3.63</td><td>3.11</td><td>3.79</td><td>3.17</td><td>4.60</td><td>4.15</td><td>4.34</td><td>4.62</td><td>3.42</td><td>2.74</td><td>3.73</td></tr><tr><td>Seedream</td><td>4.48</td><td>3.78</td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.96</td><td></td><td></td><td></td><td></td><td></td><td></td><td>3.95</td></tr><tr><td>NanoBanana-2</td><td>4.59</td><td>3.72</td><td>4.27 4.17</td><td>3.49 3.96</td><td>4.09 4.22</td><td>3.82 3.80</td><td>1.74 2.71</td><td>4.45 4.48</td><td>4.39</td><td>4.72 4.83</td><td>4.14 4.32</td><td>4.60 4.60</td><td>4.88 4.79</td><td>4.22 4.24</td><td>2.58 3.07</td><td>4.13</td></tr><tr><td>GPT-Image-2</td><td>4.67</td><td>3.93</td><td>4.34</td><td>4.02</td><td>4.32</td><td>4.08</td><td>3.65</td><td>4.67</td><td>3.41</td><td>4.86</td><td>4.25</td><td>4.67</td><td>4.99</td><td>4.38</td><td>3.11</td><td>4.22</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8 | Ecom-Bench visual scores and aggregate CTR ranking scores. VTO, Display, Wear, TLR, and Sell denote Virtual Try-On, Clothing Display, Universal Wearing, Tagline Removal, and Selling Point Display, respectively. For each of the 1,100 test cases, outputs are ranked by predicted CTR; rank-1, rank-2, and rank-3 appearances receive equal weight, and CTR reports their sum. Within the open-source and closed-source groups separately, the best result is shown in bold and the second-best is underlined. Missing results are denoted by “–”.
<table><tr><td>Model</td><td>VTO</td><td>Display</td><td>Wear</td><td>Detail</td><td>Pose</td><td>Back.</td><td>Outpaint</td><td>Extract</td><td>Text</td><td>TLR</td><td>Sell</td><td>CTR</td><td>Overall</td></tr><tr><td>Flux-2.0</td><td>4.09</td><td>3.75</td><td>2.23</td><td>3.04</td><td>3.79</td><td>3.36</td><td>4.57</td><td>3.70</td><td>2.68</td><td>3.46</td><td>3.65</td><td>293</td><td>3.48</td></tr><tr><td>Joy-Edit</td><td>1</td><td>3.61</td><td>I</td><td>2.66</td><td>4.04</td><td>3.21</td><td>3.46</td><td>3.97</td><td>3.78</td><td>2.36</td><td>4.06</td><td>=</td><td>3.46</td></tr><tr><td>Joy-Edit-Plus</td><td>4.13</td><td>3.49</td><td>2.15</td><td>2.22</td><td>3.74</td><td>2.93</td><td>3.60</td><td>2.69</td><td>2.76</td><td>3.00</td><td>3.91</td><td>406</td><td>3.15</td></tr><tr><td>LongCat</td><td></td><td>2.63</td><td></td><td>2.47</td><td>4.23</td><td>3.14</td><td>3.55</td><td>2.77</td><td>3.31</td><td>2.61</td><td>3.38</td><td></td><td>3.12</td></tr><tr><td>FireRed-Edit</td><td>4.27</td><td>2.78</td><td>1.81</td><td>1.82</td><td>4.33</td><td>3.27</td><td>4.23</td><td>3.63</td><td>3.57</td><td>3.33</td><td>3.79</td><td>362</td><td>3.35</td></tr><tr><td>Qwen-Edit-2511 KwaiMind</td><td>3.62</td><td>3.21</td><td>1.72</td><td>2.62</td><td>4.14</td><td>2.87</td><td>3.92</td><td>3.77</td><td>3.01</td><td>2.63</td><td>3.50</td><td>348</td><td>3.18</td></tr><tr><td></td><td>3.98</td><td>4.19</td><td>2.39</td><td>4.25</td><td>4.44</td><td>3.56</td><td>2.55</td><td>3.69</td><td>3.71</td><td>3.53</td><td>4.06</td><td>544</td><td>3.67</td></tr><tr><td>Seedream</td><td>4.13</td><td>4.31</td><td>2.43</td><td>3.54</td><td>4.27</td><td>3.19</td><td>2.63</td><td>3.02</td><td>3.88</td><td>2.60</td><td>4.15</td><td>525</td><td>3.47</td></tr><tr><td>NanoBanana-2</td><td>4.62</td><td>4.41</td><td>2.89</td><td>3.97</td><td>4.37</td><td>3.64</td><td>4.11</td><td>4.00</td><td>4.34</td><td>3.20</td><td>4.54</td><td>391</td><td>4.01</td></tr><tr><td>GPT-Image-2</td><td>4.64</td><td>4.57</td><td>3.06</td><td>4.44</td><td>4.60</td><td>3.84</td><td>3.53</td><td>4.28</td><td>4.38</td><td>4.10</td><td>4.75</td><td>431</td><td>4.20</td></tr></table>

The bilingual REDEdit results further show that the gain transfers across languages. KwaiMind ranks first among open-source systems on both English and Chinese overall scores, with particularly strong results for composition, extraction, portrait editing, motion, and removal. The English and Chinese scores are also close, suggesting that the bilingual instruction augmentation used during CT does not favor one language at the expense of the other. Closed-source systems remain stronger overall: GPT-Image-2 reaches 7.10 on GEdit and 4.17/4.22 on REDEdit EN/CN, while Seedream obtains the highest ImgEdit score of 4.27. The remaining gap is concentrated in several dificult categories, including viewpoint changes and some fine-grained appearance edits.

## 5.2. Ecom-Bench Results

Figure 1 summarizes the Ecom-Bench visual and CTR ranking results, while Table 8 reports the per-task visual scores and the aggregate CTR ranking score. KwaiMind achieves an overall visual score of 3.67, outperforming every open-source baseline. KwaiMind leads the open-source group on Clothing Display, Universal Wearing, Clothing Detail, Pose Change, Background Replace, Tagline Removal, and the overall score. The largest margins occur on Clothing Detail and Clothing Display, where the model must preserve local appearance or transfer garments while maintaining identity and structure. These results align with the emphasis of the SFT data mixture and the fine-grained consistency reward.

Performance is less uniform on Outpaint and Product Extract. Flux.2-dev obtains the strongest open-source Outpaint score, while Joy-Image-Edit leads Product Extract. These categories indicate that specialization does not uniformly dominate strong task-specific priors and remain important directions for further improvement. Joy-Image-Edit and LongCat do not support the multi-image inputs required by Virtual Try-On and Universal Wearing and therefore lack results for those tasks; their reported overall values are macro-averages over nine available tasks rather than all eleven and should be interpreted with this limitation.

The closed-source systems retain a clear advantage in the visual evaluation. GPT-Image-2 achieves the strongest closed-source score on ten of the eleven tasks and an overall score of 4.20, followed by NanoBanana-2 at 4.01. KwaiMind nevertheless narrows the gap on several specialized tasks: its scores on Clothing Detail, Pose Change, and Clothing Display are close to the strongest proprietary results, demonstrating that targeted data and reward design can substantially improve production-oriented capabilities without relying on a closed model.

CTR comparison. Joy-Image-Edit and LongCat support only single-image inputs, so we exclude them from the CTR comparison. We evaluate the remaining eight models on all 1,100 test cases. For each case, the model outputs are sorted by their predicted CTR scores, and appearances at rank 1, rank 2, and rank 3 are counted separately. The three counts are equally weighted and summed to form the reported CTR ranking score. As shown in Figure 1, KwaiMind records 215, 164, and 165 appearances at the three ranks, respectively, yielding the highest aggregate score of 544. This result complements the VLM-based visual score by measuring how consistently a model places among the most commercially attractive candidates under the learned preference model. Such a signal is valuable because a visually faithful edit is not necessarily an efective retail creative. Prior work on CTR-aware creative generation and product-poster optimization similarly uses behavioral feedback to guide generation beyond aesthetics alone (Fan et al., 2026; Yang et al., 2024); thus, KwaiMind’s gain indicates stronger practical potential for producing deployable commercial imagery, rather than only higher judge-based visual quality.

## 5.3. Visualization

Figures 7–9 and Figures 11–12 present qualitative comparisons on e-commerce and general image editing tasks, respectively. Each comparison shows the reference image(s), editing instruction, and outputs from KwaiMind and representative baselines, complementing the quantitative results with a direct view of instruction following, content preservation, and visual quality across the two domains. Figure 10 provides additional garment presentation examples, including flat-lay views and clothing displays on virtual models.

![](images/5b32ad15bb840a4823fddeda0d133429ccaa1f94bd2f4a6191f34bc463982626.jpg)  
Figure 7 | Qualitative comparisons on e-commerce image editing tasks (1/3).

![](images/19098fcfc06a171aaec1cbac12b148bfe9cb8af7543486a0479417b9238f7bfc.jpg)  
Figure 8 | Qualitative comparisons on e-commerce image editing tasks (2/3).

![](images/95f53fd1770df0d51ff02d0bea029dd154c0cabd78da8dfaee48eca8d5739750.jpg)  
Prompt: 将模特身上的⽩⾊卫⾐替换为图2中的浅粉⾊⻄装外套，配戗驳领、装饰绳扣、侧翻盖⼝袋和⽩⾊⽑绒袖⼝，保持模特姿势不变。

Figure 9 | Qualitative comparisons on e-commerce image editing tasks (3/3).

![](images/6aba0caf396ee62315fd75c272b0f034d99abce5eca0889fce3280461c20273b.jpg)

![](images/cce08ac6939e0439cc379fd8b26e448d210d35428f98a20e8c5b4bc9a7692817.jpg)

![](images/26cd1abdf1c02e21f1205bb37228c0207950a6ebc4c859d5e017a3f6f8930987.jpg)

![](images/b731f81bc5384329c831c0d8ee58f55451e0bb38d39914c3049293d2dab3d51d.jpg)

![](images/08005d7d7bc16e4e4d56be34aaded834154cb0ec28479a4990d76dc9906cfb8f.jpg)

![](images/3d3ec5a51509e6c4451b67484a7d7e28571e8e08b0ba30832fd380a381ef4b53.jpg)

![](images/a019f497be7a77287ef2d109cb94a672b04ab4147ed5afc3e92d0b5ac0df4154.jpg)

![](images/4872bd204fcd9364f616fc6610b617e0fee9d4f4f175b460b9cfb05b2870885d.jpg)

![](images/4afb40418f1a4acd07537b8ffeb3f8f0628ed455873dd96e69b69dd975ff7231.jpg)

![](images/4556f41475a611c33960f2b0ad7059d5568eed4a2786c357f3b38ee3af578256.jpg)

![](images/3ddf4d3c60e4ed7c548aa20b11abcfd257d1cf6b3a1ef709aac7863f2462f7d1.jpg)

![](images/fc7f2c8c962b23e5e7042a7993730483c22f13e623a3daa89416e0a5d88cac3f.jpg)

![](images/74dd1c5f597b5ff3ee42bc11411742831bed85a05f2b07afee046b4e6f2038c1.jpg)  
Reference

![](images/52b97782f07d811d766d4a20e08d5e9a82e0091f91315cb79c6f4935799fdc96.jpg)  
Clothing Detail(Tiled)

![](images/f812df5e37e04b309bc1875eca598c786a7583f28b13512e04864bb065d35271.jpg)  
Clothing Display

![](images/1eaaedf6fa8637a4bef5339478d7dada6e0aa559bcf6869e5472b53c7d95e47d.jpg)  
Clothing Display

Figure 10 | Garment presentation examples from KwaiMind, including tiled views and clothing displays on virtual models.

KwaiMind

Prompt: Transfer the image into a Lego-brick stop-motion diorama style.

![](images/0a512c1a09e8e5f7d605eb246563a26d4d217c89d4530ec916c8c369561dfcb6.jpg)  
Reference

![](images/8e5d8e82cdc4954c2b1a95810b03af9de2c7342aecea11d5ab97edade05666bc.jpg)  
KwaiMind

![](images/d2b89fb445f6ba789bd21042cab2d5a064dfba985cd14f4eb525e27958269c16.jpg)  
Qwen-Image-Edit-2511

![](images/4d641f00ed0cc08ec37dd14690c7dc7760ab9d119235900f12ca73ee9c81afcb.jpg)  
FireRed-Image-Edit

![](images/bc995d793bbf05e89f5b43a3b81c9b4d606de9465ec1921d0959e43ef9b98d0c.jpg)  
Seedream-5.0

![](images/ced193e7fb2cfb353c3c7c4419cc9f8b6b75d3c1c67ecdb48696d302783bef61.jpg)  
GPT-Image-2

Prompt: Change the black base of the barbecue grill to red.  
![](images/e52bcedc8b6008ab1c637b7f15c8c99dd53426c35bd1410ae1f5259babfdc048.jpg)  
Reference

![](images/87eda712dc2a981f4add5c826b0c187b384df21d8608fc877de4cb8666fd2db7.jpg)  
KwaiMind

![](images/da6472ca2fb8367d9774bec7669f41fba4343deb5e5b1ac4950951684b43bddb.jpg)  
Qwen-Image-Edit-2511

![](images/fc59a8efc6882b1e4fa1bf4134bc1d7a3ef5c897be8ce4329443b17204dfab13.jpg)  
FireRed-Image-Edit

![](images/18ab4bcd41932b1cd01b0e8f5ff5a59107496ce925e8b8d440e15e3c3169cdef.jpg)  
Seedream-5.0

![](images/bc1fb20fb35f2f43c782a65902cdd3db56967ab45fe1a404aea8d92ded1ff48b.jpg)  
GPT-Image-2

Prompt: Remove the footprint in the sand.

![](images/bb408b98d8c926be58321c6b8752d31f594cf03a179d39a10409e50dc5657d4a.jpg)  
Reference

![](images/49973c7ba1752ff3a5844a9da938f07cc5a4ba5b394647bdc779066cbe03e2a7.jpg)  
KwaiMind

![](images/a4e093d8bdc5fdca1787a784c8735a0ee90315a29c03389960924049fe11c974.jpg)  
Qwen-Image-Edit-2511

![](images/c2259f4e887775c1c9ee4aed32640bbc4c46a687a38368296b983c44ebc20b66.jpg)  
FireRed-Image-Edit

![](images/6896ba410863e7b5a999e3378ec05315f9a68e1d206804ef98512d83d878a282.jpg)  
Seedream-5.0

![](images/7715bb14f6c6cc85e53271c451fd52145576604369cf5a8db1efc967226869e6.jpg)  
GPT-Image-2

Prompt: Extract the man wearing a light blue plaid suit, red-striped tie, and glasses standing in front of the railway track.  
![](images/5fc7e4a98a361a962e023e918719e1961cd225c4db6a893471d670469a92fa34.jpg)  
Reference

![](images/f41ad494a22949b71f932caa3be03ef514f46dfd665b835b3976053e4e796d4a.jpg)

![](images/ed3dd12a79569ce0c69f50105d1144443fa5acddde834928501c30fc5701117d.jpg)  
Qwen-Image-Edit-2511

![](images/cb4354bbcd14786d94dfa8383429d49c5d802d645dc2b0586d46cb9e75c652c5.jpg)  
FireRed-Image-Edit

![](images/659d2d799527f1dfb1fbdb659f2082eaa8a349d225b8fc28ca6db9f41501c8bd.jpg)  
Seedream-5.0

![](images/b1c2b3a962f2fba41123a1bdb1c5231e127accf8c61ca2a11494c15b4f4507a1.jpg)  
GPT-Image-2

Prompt: Change the text 'TAX TIME' on the sticky note to 'Deadline是第⼀⽣产⼒' (Deadline is the primary productive force), keeping the original handwritten style.

Reference

![](images/c32af350b0796ded4939ad7cf33005854efbdf1b7fbabf9850427d75fe3454be.jpg)

![](images/9bc42d716cb7a2f278665b07c3a595db116f256f8fd390e3c27b7dfe4655f70a.jpg)  
KwaiMind

![](images/0793684bb478067ce9879f09e16e6f75145997f63455c90ac5a26ad232643d13.jpg)  
Qwen-Image-Edit-2511  
FireRed-Image-Edit

![](images/eb75899508dc34eaf73cb8efdf0205631b847a421c951346a1f3728d2c5e968d.jpg)  
Seedream-5.0

![](images/ad33a5b893809f6af2d4b83ecdd5854b16291e221fb9b958c7aa9aef5139f269.jpg)  
Figure 11 | Qualitative comparisons on general image editing tasks (1/2).

![](images/b085e574c2cc23f7566d2f50f31e873d66e858e4cba7ed88b916f9f76f96b8a0.jpg)  
GPT-Image-2

Prompt: 将背景改为森林

![](images/519382323ffd3d9c33a74ee67b259807242df159031e77bf44cf20900142f132.jpg)  
Reference

![](images/80f5aec72ac52f5823523e58f93e5975ab137c06b1019845f92234fcae798586.jpg)  
KwaiMind

![](images/045c71739097aa9c3240f49e5bc4e3464cd17cf34f1a045b0b25ca40f20825bf.jpg)  
Qwen-Image-Edit-2511

![](images/bc47c918bd208aa27974102e2f2da4a473b3b611a00ddd1a5b0c38251e7108f5.jpg)  
FireRed-Image-Edit

![](images/06df1c2ca9ceb7496d141ac0a9f332a3075c7327875ea3b4b6a4f30010aee64c.jpg)  
Seedream-5.0

![](images/db2c7d88f1514f0627e1a21778c81cbe9e91abb223a45c6a3fee140d5162f36d.jpg)  
GPT-Image-2  
Prompt: Change the person’s shirt color to blue.

![](images/bd8c3a4ce9081881697095bff0fbf4c200b11fc9850bc15d0cf822977e5095ad.jpg)  
Reference

![](images/be37b9d48b1497c7dcc56955449f49408434d8f0bc195e569203d8b9cc78e407.jpg)  
KwaiMind

![](images/b2fae169464150a79c43f6940dccf004d02bec750dace46e43ebbab4a61bc62a.jpg)  
Qwen-Image-Edit-2511

![](images/7cf3158a7d9df8c917bf514b4953a1477639ed3769550b126d626fbdbcaf35c0.jpg)  
FireRed-Image-Edit

![](images/5da7038a378654acf46c82a4e2fb019f2562fcfbd8b130ad610c5b5393e42eec.jpg)  
Seedream-5.0

![](images/bf08ca63c342ad0e28b19871bffd0462ef6021abeee86c3233d986e55987e536.jpg)  
GPT-Image-2

Prompt: Add a cat sitting on the floor near the bottom right corner.

![](images/d67cd3369db6a5a7630d84cfa5e9031f26d8087940c6c64a4fbb2dc8df6d23a7.jpg)  
Reference

![](images/3cd4defdbf0057f06fae89064c0c9170978652a8f62841231c23242bd8bc322a.jpg)  
KwaiMind

![](images/ef687e42deeeb1e6a65d7f27fca5c4e9991d98ae28590139c222a6b1a4c7462d.jpg)  
Qwen-Image-Edit-2511

![](images/dc56889d345703321a1cc40a07fb275f86993c3aa82433dc4329eaa46e520cea.jpg)  
FireRed-Image-Edit

![](images/bafce26860dbfe5dc6550f4ba0c82d4331850ddd86f48488b64dbf9a881cb2a4.jpg)  
Seedream-5.0

![](images/8473bcc41d5bca11640e61b641b2f441c91285c7413bcbf82fe6b3b5b07ee2b9.jpg)  
GPT-Image-2

Prompt: Remove the table and the flower vase in the foreground.

![](images/bfcece124bf2ea20cbc65003aebfae1d13e8e81bd013bd8dd8d3548267aef3c6.jpg)  
Reference

![](images/4f3ded42f98ccf8584a869763d4f29114c5a74c7bb0a8e188c34f2f9c4de139c.jpg)  
KwaiMind

![](images/21182437fe234eb9ba6725402cd91071e50fe2215469e8e4faa3b8d51a9ef587.jpg)  
Qwen-Image-Edit-2511

![](images/ede84fe2332dcdcb64a0d2d6ea4db459885aee9e987acee9a14a7c9d5f672404.jpg)  
FireRed-Image-Edit

![](images/51731a59638c63d388b2fee991605e784e04718311f69e113313bf12f063ff54.jpg)  
Seedream-5.0

![](images/707552a7f1cb9bed9b8fef903573260bf5341ad3f8544c160177e00e87f9f228.jpg)  
GPT-Image-2

Prompt: Make the person in the picture appear in a friendly greeting state, with the arm raised naturally, palm open, fingers slightly curled, and posture relaxed.

![](images/d84b4880f8dba5070d167ea583fbbe087b21d2561ef571bc34f3724535ac9c7b.jpg)  
Reference

![](images/bd50de5448b83d46b5a7d62970bf0b0651779746a4b6950f3e8cf21811af27c8.jpg)  
KwaiMind

![](images/ab127d5db7ec4786154e82dc4622c30ddda07f0ef028b1498527abe50891eb7c.jpg)  
Qwen-Image-Edit-2511  
FireRed-Image-Edit

Figure 12 | Qualitative comparisons on general image editing tasks (2/2).

![](images/9c6a58d5935e8585af43935c9ae9177b2986f782cf7a4261aa628d4819130fde.jpg)

![](images/67f0d999fe090f805d0e3ff1ebe558337bb5ba954f28629c233fcd06068d4e8f.jpg)  
Seedream-5.0

![](images/104db8cf995ece3d242fe512448ce06c8dd892ae77c50a459356b3c03159168f.jpg)  
GPT-Image-2

## 6. Conclusion

We present KwaiMind, an image editing system that combines broad editing competence with the requirements of e-commerce content production through an agent-based data engine, staged training, and specialized reward optimization. Ecom-Bench complements general benchmarks with taskspecific visual evaluation and CTR-based ranking. KwaiMind achieves the strongest aggregate results among the evaluated open-source editors across these benchmarks, while CTR-guided optimization and online material selection demonstrate practical commercial value. These findings highlight the benefit of aligning data, training objectives, and evaluation with real production needs. Future work will focus on closing the visual-quality gap with proprietary systems and improving robustness on compositional and multi-reference edits.

## Contribution

Core Contributors (listed alphabetically): Boheng Zhang, Fan Yang, Jia Sun, Junlong Wu, Wenwu Ou, Yuting Hu, Zijun Li

Major Contributors (listed alphabetically): Dewen Fan, Fei Zuo, Honglie Wang, Huaiqing Wang, Pengcheng Wei, Yimin Zhou

Support Contributors (listed alphabetically): Haixuan Gao, Lihui Peng, Tingxuan She, Yuqing Li

## References

S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine. Training difusion models with reinforcement learning. In International Conference on Learning Representations, volume 2024, pages 4965–4987, 2024.

Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.

T. Brooks, A. Holynski, and A. A. Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392– 18402, 2023.

C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, and G. Hullender. Learning to rank using gradient descent. In Proceedings of the 22nd international conference on Machine learning, pages 89–96, 2005.

Z. Chang, S. Weng, H. Ouyang, Y. Hong, L. Lin, S. Li, and B. Shi. L-vocal: Language-based video colorization with audio alignment. International Journal of Computer Vision, 134(5):208, 2026.

G. Chen, E. Cui, C. Tian, D. Yang, G. Yang, Y. Qiao, H. Li, G. Luo, and H. Zhang. Scaleedit-12m: Scaling open-source image editing data generation via multi-agent framework. arXiv preprint arXiv:2603.20644, 2026.

X. Chen, W. Feng, Z. Du, W. Wang, Y. Chen, H. Wang, L. Liu, Y. Li, J. Zhao, Y. Li, et al. Ctr-driven advertising image generation with multimodal large language models. In Proceedings of the ACM on Web Conference 2025, pages 2262–2275, 2025.

Y. Choi, S. Kwak, K. Lee, H. Choi, and J. Shin. Improving difusion models for authentic virtual try-on in the wild. In European Conference on Computer Vision, pages 206–235. Springer, 2024.

P. Esser, S. Kulal, A. Blattmann, R. Entezari, J. Müller, H. Saini, Y. Levi, D. Lorenz, A. Sauer, F. Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning (ICML), 2024.

J. Fan, Y. Qin, W. Feng, Y. Chen, Y. Li, A. Ma, Y. Li, L. Zhuang, H. Bian, Z. Zhang, et al. Autopp: Towards automated product poster generation and optimization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 3768–3776, 2026. Issue 5.

N. Gong, Z. Li, S. Dong, H. Bai, W. Ying, X. Wang, and Y. Fu. Sculpting features from noise: Reward-guided hierarchical difusion for task-optimal feature transformation. In Advances in Neural Information Processing Systems, volume 38, pages 23452–23474, 2026.

Y. Heng, C. Jiang, H. Yang, S. Zhang, and W. Ye. Eve: Verifiable self-evolution of mllms via executable visual transformations. arXiv preprint arXiv:2604.18320, 2026.

C. Jiang, Y. Heng, W. Ye, H. Xu, M. Yan, J. Zhang, F. Huang, and S. Zhang. Vlm-r<sup>3</sup>: Region recognition, reasoning, and refinement for enhanced multimodal chain-of-thought. In Advances in Neural Information Processing Systems, volume 38, pages 63841–63869, 2026.

M. Ku, D. Jiang, C. Wei, X. Yue, and W. Chen. VIEScore: Towards explainable metrics for conditional image synthesis evaluation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12268–12290. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.663.

B. F. Labs, S. Batifol, A. Blattmann, F. Boesel, S. Consul, C. Diagne, T. Dockhorn, J. English, Z. English, P. Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Q. Li, J. Yu, K. Jiang, Y. Wei, Z. Xing, P. Li, R. Chu, S. Zhang, Y. Liu, and Z. Wu. Difusionopd: A unified perspective of on-policy distillation in difusion models. arXiv preprint arXiv:2605.15055, 2026a.

Z. Li, H. Yan, S. Li, K. Luo, L. Lu, X. Yang, and W. Lin. Difpcn: Latent difusion model based on multi-view depth images for point cloud completion. arXiv preprint arXiv:2509.23723, 2025.

Z. Li, Y. Zhou, J. Sun, H. Wang, P. Wei, J. Wu, Y. Heng, J. Wang, H. Ouyang, B. Zhang, et al. Detailanywhere: Fashion detail generation via cross-modal feature alignment distillation. arXiv preprint arXiv:2607.02220, 2026b.

J. Lin, J. Wu, F. Zuo, H. Ouyang, D. Fan, B. Zhang, H. Wang, J. Sun, F. Yang, H. Liu, et al. Joint alignment and distillation for video generation via sample-guided distribution matching. arXiv preprint arXiv:2609.04283, 2026.

Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In International Conference on Learning Representations (ICLR), 2023.

S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li, J. Yang, H. Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pages 38–55. Springer, 2024.

S. Liu, Y. Han, P. Xing, F. Yin, R. Wang, W. Cheng, J. Liao, Y. Wang, H. Fu, C. Han, G. Li, Y. Peng, Q. Sun, J. Wu, Y. Cai, Z. Ge, R. Ming, L. Xia, X. Zeng, Y. Zhu, B. Jiao, X. Zhang, G. Yu, and D. Jiang. Step1X-Edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

J. Ma, X. Zhu, Z. Pan, Q. Peng, X. Guo, C. Chen, and H. Lu. X2edit: Revisiting arbitrary-instruction image editing through self-constructed data and task-aware representation learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 7764–7772, 2026. Issue 10.

A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegrefe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.

Meituan LongCat Team, H. Ma, H. Tan, J. Huang, et al. Longcat-image technical report. arXiv preprint arXiv:2512.07584, 2025.

D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach. Sdxl: Improving latent difusion models for high-resolution image synthesis. In International Conference on Learning Representations, volume 2024, pages 1862–1874, 2024.

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, et al. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, volume 2025, pages 28085–28128, 2025.

R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-resolution image synthesis with latent difusion models. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 10674–10685. ieee, 2022.

V. Singla, K. Yue, S. Paul, R. Shirkavand, M. Jayawardhana, A. Ganjdanesh, H. Huang, A. Bhatele, G. Somepalli, and T. Goldstein. From pixels to prose: A large dataset of dense image captions. arXiv preprint arXiv:2406.10328, 2024.

H. Song, H. Su, I. Shalyminov, J. Cai, and S. Mansour. Finesure: Fine-grained summarization evaluation using llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 906–922, 2024.

L. Song, W. Li, G. Ma, W. Tang, B. Wang, Y. Zhang, et al. Joyai-image: Awaking spatial intelligence in unified multimodal understanding and generation. arXiv preprint arXiv:2605.04128, 2026.

S. I. Team, C. Qiao, C. Hui, C. Li, C. Wang, D. Song, J. Zhang, J. Li, Q. Xiang, R. Wang, et al. Firered-image-edit-1.0 technical report. arXiv preprint arXiv:2602.13344, 2026.

M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Y. Tuo, W. Xiang, J.-Y. He, Y. Geng, and X. Xie. Anytext: Multilingual visual text generation and editing. In International Conference on Learning Representations, volume 2024, pages 56783–56799, 2024.

B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik. Difusion model alignment using direct preference optimization. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8228–8238. IEEE, 2024.

H. Wang, J. Sun, Z. Li, J. Wu, P. Wei, J. Wang, Y. Heng, B. Zhang, H. Wang, D. Fan, et al. Textrefine: Improving textual fidelity, spatial placement, and glyph rendering for text editing in product posters. arXiv preprint arXiv:2608.19637, 2026a.

J. Wang, C. Lin, L. Sun, Z. Cao, Y. Yin, L. Nie, Z. Yuan, X. Chu, Y. Wei, K. Liao, et al. Geometry-guided reinforcement learning for multi-view consistent 3d scene editing. arXiv preprint arXiv:2603.03143, 2026b.

J. Wang, H. Ouyang, J. Lin, C. Lin, D. Fan, B. Zhang, H. Fan, F. Zuo, J. Sun, H. Wang, et al. Cac: Advancing video reward models via hierarchical spatiotemporal concentrating. arXiv preprint arXiv:2605.11723, 2026c.

S. Wang, Q. Liu, T. Ge, D. Lian, and Z. Zhang. A hybrid bandit model with visual priors for creative ranking in display advertising. In Proceedings of the web conference 2021, pages 2324–2334, 2021.

X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

C. Wei, Z. Xiong, W. Ren, X. Du, G. Zhang, and W. Chen. Omniedit: Building image editing generalist models through specialist supervision. In International Conference on Learning Representations, 2025.

C. Wu, J. Li, J. Zhou, J. Lin, K. Gao, K. Yan, S.-m. Yin, S. Bai, X. Xu, Y. Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

J. Wu, Y. Cheng, H. Liu, and H. Liu. Arc: Robots adaptive risk-aware robust control via distributional reinforcement learning. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 10656–10663. IEEE, 2025b.

J. Wu, J. Lin, J. Sun, B. Zhang, H. Wang, D. Fan, H. Liu, Q. Gan, F. Yang, and T. Gao. Step back to move forward: Reflection-aware preference optimization for visual generation. arXiv preprint arXiv:2609.04282, 2026.

T.-H. P. Wu, H. Lee, J. Ge, J. Gonzalez, T. Darrell, and D. Chan. Generate, but verify: Reducing hallucination in vision-language models with retrospective resampling. Advances in Neural Information Processing Systems, 38:65749–65777, 2025c.

B. Xia, Y. Zhang, J. Li, C. Wang, Y. Wang, X. Wu, B. Yu, and J. Jia. Dreamomni: Unified image generation and editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28533–28543, 2025a.

Y. Xia, Y. Zhou, J. Wang, B. An, H. Wang, Y. Wang, and B. Chen. Difpc: Difusion-based high perceptual fidelity image compression with semantic refinement. In International Conference on Learning Representations, volume 2025, pages 102324–102350, 2025b.

Y. Xia, Y. Zhou, J. Wang, M. Hong, H. Wang, B. Chen, and Y. Wang. Diric: Difusion prior refinement for eficient low-rate image compression. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

H. Yang, J. Yuan, S. Yang, L. Xu, S. Yuan, and Y. Zeng. A new creative generation pipeline for click-through rate with stable difusion model. arXiv preprint arXiv:2401.10934, 2024.

J. Yang, H. Zhang, F. Li, X. Zou, C. Li, and J. Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023a.

Z. Yang, A. Zeng, C. Yuan, and Y. Li. Efective whole-body pose estimation with two-stages distillation. In 2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 4212– 4222. IEEE, 2023b.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Y. Ye, X. He, Z. Li, B. Lin, S. Yuan, Z. Yan, B. Hou, and L. Yuan. ImgEdit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025.

Q. Yu, W. Chow, Z. Yue, K. Pan, Y. Wu, X. Wan, J. Li, S. Tang, H. Zhang, and Y. Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26125–26135, 2025.

R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

Z. Zhang, Q. Dai, X. Bo, C. Ma, R. Li, X. Chen, J. Zhu, Z. Dong, and J.-R. Wen. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43 (6):1–47, 2025.

K. Zheng, H. Chen, H. Ye, H. Wang, Q. Zhang, K. Jiang, H. Su, S. Ermon, J. Zhu, and M.-Y. Liu. Difusionnft: Online difusion reinforcement with forward process. In International Conference on Learning Representations, volume 2026, pages 134129–134150, 2026.

L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623, 2023.

P. Zheng, D. Gao, D.-P. Fan, L. Liu, J. Laaksonen, W. Ouyang, and N. Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CAAI Artificial Intelligence Research, 2024.