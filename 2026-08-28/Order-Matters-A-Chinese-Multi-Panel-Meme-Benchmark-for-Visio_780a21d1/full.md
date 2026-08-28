# Order Matters: A Chinese Multi-Panel Meme Benchmark for Vision-Language Reasoning

Haihan Li<sup>1</sup>, Haihao Li<sup>2</sup>, Zhenfei Xu<sup>1</sup>, and Jize Qian<sup>1</sup>

1 Shanghai Maritime University, Shanghai, China 2 Dalian Maritime University, Dalian, China

Abstract. Many multimodal tasks depend on how visual elements are ordered and composed, not only on recognizing them in isolation. Internet memes are a compact case of this problem: their punchline often depends on a constrained reading order and cross-panel visual–textual cues. While large vision-language models (LVLMs) show strong performance on single-image understanding, it remains unclear whether they can perform sequence-aware reasoning over structured meme layouts, especially in Chinese social media. We introduce CMPM, a Chinese Multi-Panel Meme benchmark with 1,214 annotated samples covering five structural types, ordering dependency, panel-order constraints, and optional comment context. We formulate a two-layer evaluation: Task 1 probes structure typing and order-sensitive panel sequencing (with a context ablation setting), and Task 2 evaluates Chinese meme explanation generation with human ratings on five 1–3 Likert dimensions (visual, panel, humor, context, and faithfulness). We benchmark five representative LVLMs under a unified protocol. Results indicate that canonicaldisplay accuracy is not by itself evidence of order understanding: the primary shufled condition produces a sharp accuracy drop, revealing a persistent gap in order-sensitive multimodal reasoning. Task 2 preferences place Gemini 3.1 Pro and GPT-5.5 above the open models, while comment context yields only a small and mixed Core4 gain. Code and data will be released upon acceptance.

Keywords: Multi-panel memes · Multimodal reasoning · Human evaluation

## 1 Introduction

Multimodal understanding often depends on how visual elements are arranged, sequenced, and jointly interpreted, not just on recognizing objects or text in isolation. Memes are a pervasive form of multimodal communication that combine images, overlaid text, and cultural references to convey humor, irony, or social commentary [14, 7]. Beyond the classic single-image template, Chinese-language social platforms commonly circulate multi-panel memes—comic-strip layouts, chat screenshots, and comparison grids—in which meaning emerges from the sequence of panels and their inter-panel relations. Reordering panels can destroy the punchline, invert a contrast, or break a conversational turn, making reading order a first-class variable for multimodal understanding.

Large vision-language models (LVLMs) such as LLaVA [8], Qwen-VL [1], InternVL [3], and proprietary systems have advanced image captioning, visual question answering, and document understanding. Existing meme benchmarks primarily target classification, hate/ofense detection, or single-image explanation [7, 15, 20]. Many of these tasks can be solved through local cue matching: recognizing salient people, objects, or text and associating them with a label or likely interpretation without reconstructing global panel relations. They therefore need not test setup-to-punchline dependencies or distinguish a valid reading order from a shufle. Text-only Chinese meme resources such as CHIME [18] evaluate cultural phrase understanding, but do not address panel structure or visual order. To our knowledge, existing resources do not provide a dedicated testbed for order-sensitive multimodal reasoning on Chinese multi-panel memes.

![](images/87e271f2da1b284c899038759d42e92444aac84522d752b48f26fd7e428d9f0c.jpg)  
Fig. 1. Order matters in multi-panel memes. In the canonical order, P1 establishes a truth-versus-lie premise, P2–P3 form an interrupted conversational transition, and P4 delivers the service-style punchline “Customer first.” The shufled presentation preserves the local visual content but breaks this setup-to-punchline progression, illustrating why local cue matching is insuficient for global meme understanding. The Chinese dialogue reads: P1, “One of us can only lie forever, and the other can only tell the truth forever”; P2, “This...”; P3, “Oops!”; P4, “Customer first.”

We introduce CMPM (Chinese Multi-Panel Memes), a benchmark designed for sequence-aware vision-language evaluation. Each sample provides cropped panels, structural type labels, ordering dependency, order-group constraints, and optional title/comment context. Building on this resource, we define a two-layer protocol: Task 1 asks models to recover structure and reading order under controlled display and context conditions; Task 2 asks models to generate Chinese explanations that are scored by human raters on five 1–3 ordinal dimensions (visual correctness, panel coherence, humor comprehension, context handling, and faithfulness), providing a human-evaluation layer complementary to automatic LVLM metrics.

Our main contributions are:

– We introduce CMPM, a Chinese multi-panel meme benchmark with 1,214 samples and a fine-grained annotation schema that jointly represents structure, order, and context relations.

– We design a two-layer, order-sensitive evaluation protocol spanning Task 1 panel sequencing and Task 2 human explanation ratings, including a controlled comment-aware context ablation setting (CAS).

– Our evaluation shows that current LVLMs remain brittle under shufled panel order, while post context yields a small average gain with both helpful and harmful instance-level efects.

## 2 Related Work

Meme and humor datasets. Early meme corpora emphasize hate speech, ofense, and sentiment in primarily English single-image settings [7, 15, 14]. MemeRea-Con [20] evaluates contextual meme understanding with post text and comments, but does not treat Chinese multi-panel sequence structure as the primary variable. CHIME [18] targets phrase-based Chinese Internet memes for text-only LLM evaluation and is complementary to our visual multi-panel setting. Table 1 summarizes the positioning of CMPM against representative resources.

Sequential multimodal understanding. Order-sensitive reasoning also appears in comic understanding, visual narrative modeling, and document understanding, where meaning depends on the reading order of panels, speech bubbles, or layout regions. COMICS studies closure-driven inference across comic panels [6]; CoMix includes reading order within a broader multi-task comic benchmark [16]; and CHROMIC directly evaluates panel and description reordering for chronological reasoning [5]. ImageChain instead models image sequences as multi-turn conversations for next-scene description [13]. These resources primarily target narrative continuation, chronological ordering, OCR-grounded reading order, or layout-level structure. CMPM isolates a diferent combination: order-sensitive interpretation of culturally grounded Chinese memes, where panel order interacts with punchlines, contrast, dialogue, and optional post context. CMPM is therefore complementary to sequential multimodal benchmarks rather than a duplicate of them.

LVLM benchmarks and multimedia reasoning. LVLMs unify visual encoders with large language models [8, 1, 3, 17]. General suites such as MMBench [9] and MM-Star [2], as well as fine-grained document benchmarks such as MMDocBench [21], stress perception and grounding, yet rarely isolate multi-panel meme order as a controlled factor. Related comic-strip and document-understanding settings likewise treat panels as spatially organized units, but generally do not isolate meme-specific reading-order recovery under controlled shufling. CMPM contributes a focused multimedia modeling setting: sequence-aware understanding of culturally grounded visual memes.

Table 1. Comparison with representative meme / multimodal resources. Context: use of post text or comments. HE: human-evaluated explanations. OS: order-sensitive panel sequencing.
<table><tr><td>Resource</td><td>Lang. Multi-panel OS Context HE Scale</td><td></td><td></td><td></td><td></td></tr><tr><td>Hateful Memes [7]</td><td>EN</td><td></td><td rowspan="2"></td><td></td><td>~10k</td></tr><tr><td>MemeReaCon [20]</td><td>EN</td><td> $\checkmark ^ { * }$ </td><td></td><td>1,565</td></tr><tr><td>CHIME [18]</td><td>ZH</td><td></td><td></td><td> $\checkmark ^ { \dagger }$ </td><td>1.5k</td></tr><tr><td>CMPM (ours)</td><td>ZH</td><td>√ √</td><td>√</td><td></td><td>√ 1,214</td></tr></table>

<sup>∗</sup>Post context for intent, not CMPM-style panel-order ablation. $\mathrm { ^ { \dag } T e x t - o n l y }$ meme explanation (no visual panels).

Table 2. CMPM task taxonomy. Primary order pool: gold-strong non-parallel samples with order groups.
<table><tr><td>Task</td><td>Sub-task</td><td>Setting  $/ \ n$ </td><td>Metrics</td></tr><tr><td rowspan="3">Task 1 typing</td><td>Structure</td><td>Canonical, no comments;  $N { = } 1 2 1 4$ </td><td> $\operatorname { A c c . }$  2 macro  ${ \bf \nabla } \cdot / { \bf w } { - } F _ { 1 }$ </td></tr><tr><td></td><td>Order recovery Shuffled, no comments; n=868</td><td>Group-aware order Acc.</td></tr><tr><td>CAS</td><td>2×2 display×context; ≈252/cell</td><td>Order Acc.</td></tr><tr><td></td><td></td><td>Task 2 Explanation + Main 64 full-context memes; paired human ratings ablation on 31 matched pairs; 3 raters</td><td>Mean Likert (5 dims, 1–3)</td></tr></table>

Explanation and human evaluation. Task 2 complements automatic LVLM metrics with human ratings of free-form Chinese explanations.

## 3 Problem Definition

Let a multi-panel meme be an ordered sequence $P = ( p _ { 1 } , \ldots , p _ { n } ) ( n \geq 2 )$ with optional post context C. Gold labels include type $y _ { \mathrm { t y p e } } .$ ordering dependency $y _ { \mathrm { o d } } \in$ {strong, weak}, and order groups $G ;$ a canonical order $\pi ^ { \star }$ is one valid linear extension of the precedence constraints encoded by G. Table 2 summarizes the task taxonomy.

## 3.1 Task 1: Structure and Order Understanding

Task 1 evaluates discriminative multimodal understanding under controlled presentations.

Structure typing. Given canonical-order panels with comments hidden, the model predicts $\hat { y } _ { \mathrm { t y p e } }$ among narrative, progressive, conversational, parallel, and comparison. We report accuracy and macro-/weighted- $F _ { 1 }$

Order-sensitive sequencing. For samples with $y _ { \mathrm { o d } } = \mathrm { s t r o n g }$ (and non-parallel structure with available $G )$ , panels are displayed under a fixed shufle. The model predicts a reading order πˆ over the currently displayed panel indices. The primary metric is group-aware logical-order accuracy: πˆ is correct if it is a valid linear extension of the gold precedence constraints after the canonical order is mapped to display indices; within-group permutations are allowed.

Context ablation setting (CAS). On a comment-available subset, we cross display (correct $/$ shuffled) and context (no\_context / with\_comments). CAS measures whether comment text helps or hurts order recovery.

## 3.2 Task 2: Explanation Generation and Human Evaluation

Task 2 evaluates Chinese meme explanation generation with human judgments. Given the same meme with or without post context, a model produces a Chinese explanation eˆ. Three raters, blinded to model identity, score each explanation separately: a rating page shows one meme and one anonymized explanation, scored on five 1–3 Likert dimensions (visual correctness, panel coherence, humor comprehension, context handling, and faithfulness). The primary aggregate for condition comparisons is Core4, defined as the mean of Visual, Panel, Humor, and Faithfulness; Context is excluded to avoid image-only scoring bias. On a distinct comparison page, raters holistically order five anonymous explanations; this judgment is independent of scalar scores and permits ties for similarly good or similarly poor explanations. We convert each ordering into pairwise wins and ties. We report scalar means on 64 context-present memes (3 raters) and a paired ablation on 31 matched context-present/context-absent pairs.

## 4 The CMPM Dataset

## 4.1 Collection and Inclusion

CMPM contains 1,214 Chinese multi-panel memes collected from major social platforms, including Weibo, Bilibili, Xiaohongshu, Tieba, and Douyin. Each sample is stored with source metadata when available, cropped panel images $( n \geq 2 )$ , and optional textual context. The main set keeps only multi-panel samples with $n \geq 2 ;$ single-image cases, including complex collages, are excluded from the main evaluation scope. The collection pipeline filters raw posts into multi-panel candidates, crops panel images, annotates structure/order/context labels, and adjudicates gold labels used to build the Task 1 and Task 2 evaluation manifests.

![](images/658f353f9940c9ee6c36d5642504f9f2d8a894d4d7c3146018e84cb8f11d347e.jpg)  
Fig. 2. Five primary structure types in CMPM with representative multi-panel examples (narrative, progressive, conversational, comparison, and parallel).

## 4.2 Annotation Schema

Annotators label each meme with the following core fields:

Primary structure type $y _ { \mathrm { t y p e } }$ . One of five types selected by a decision tree prioritizing global layout over local cues: parallel (juxtaposition/analogy), progressive (degree intensification), conversational (dialogue/chat turns), comparison (contrast/reversal), and narrative (event or causal sequence).

Ordering dependency $y _ { \mathrm { o d } }$ . Operationalized by a shufle test: strong if reordering makes the meme incomprehensible or removes the punchline; weak otherwise.

Order groups G. Groups encode block precedence while allowing within-group permutations (e.g., [[1], [2, 3], [4]]).

Context–meme relation. When efective text context exists, annotators label relations such as image-dominant, text-dominant, complementary, sarcastic conflict, or unclear.

## 4.3 Corpus Statistics

Table 3 reports the gold-label distribution after adjudication. Panel counts range from 2 to 15 (411 / 257 / 332 / 129 / 47 samples for 2–6 panels; 38 samples with $\geq 7$ panels). Here, textual context means any non-empty title/caption/comment field attached to the post. Among 1,214 memes, 322 have comment/danmakustyle context usable for CAS, of which 252 also belong to the order pool used in the context ablation. We additionally record 991 samples with any non-empty textual field, but do not treat this broader count as equivalent to comment availability.

Table 3. CMPM corpus statistics under final gold labels (N = 1214).
<table><tr><td>Type</td><td>n</td><td>%</td><td>Ordering</td><td>n</td><td>%</td></tr><tr><td>comparison</td><td>373</td><td>30.7</td><td>strong</td><td>868</td><td>71.5</td></tr><tr><td>narrative</td><td>319</td><td>26.3</td><td>weak</td><td>346</td><td>28.5</td></tr><tr><td>conversational</td><td>285</td><td>23.5</td><td></td><td></td><td></td></tr><tr><td>parallel</td><td>137</td><td>11.3</td><td></td><td></td><td></td></tr><tr><td>progressive</td><td>100</td><td>8.2</td><td></td><td></td><td></td></tr></table>

## 4.4 Annotation Quality

Three annotators independently labeled all 1,214 samples under a unified guideline. Disagreements were resolved by discussion to produce final gold labels used in all experiments. Agreement is high for structure type (Fleiss κ = 0.898) and ordering dependency (Fleiss κ = 0.875). Order-group exact match exceeds 0.97 among non-parallel samples. Context–meme relation is more subjective (Fleiss $\kappa = 0 . 4 9 9 )$ , which we treat as an auxiliary label.

## 5 Evaluation Protocol

## 5.1 Models

We evaluate five LVLMs under a shared manifest so that display order, context strings, and prompts are model-agnostic:

– Open-source: InternVL3.5-8B [11], Qwen3.5-9B [12], and GLM-4.1V-9B-Thinking [19];

– Closed-source: GPT-5.5 [10] and Gemini 3.1 Pro [4].

We do not fine-tune any model on CMPM. Closed-source models are evaluated under the same final shared manifest as open models.

## 5.2 Task 1 Protocol Details

Type prediction. One request per meme: panels in canonical order, no comments (N = 1214).

Order prediction. The order pool contains gold-strong non-parallel samples with valid order groups (N = 868). The primary leaderboard condition is shufled display + no comments (shared seed = 42). Hard cases with ≥ 7 panels are retained and tagged separately. The generation manifest has 1,624 unique requests per model after merging primary and CAS conditions.

CAS. On 252 comment-available order-pool memes, we evaluate the 2 × 2 grid of display × context and report group-aware logical-order accuracy for each condition in Table 6.

Metrics. For ordering we report group-aware logical-order accuracy on parseable outputs. For typing we report accuracy, macro- $F _ { 1 }$ , and weighted- ${ \bf \nabla } \cdot { \cal F } _ { 1 }$ , plus parse rate.

## 5.3 Task 2 Protocol Details

Models generate explanations with or without available post text. Scalar scoring uses randomized blind pages, each showing one meme and one anonymized candidate; a separate randomized comparison page shows five candidates for holistic ranking. Three independent non-author annotators received common rubric training and independently completed both judgments with model identities and other ratings hidden. Holistic ties are allowed for similarly good or similarly poor explanations and are expanded into ten pairwise wins and ties. The main aggregate contains 64 unique context-present memes selected from a stratified 400-meme generation pool (12–13 per structure type), balancing type, panel count, context availability, and annotation cost. The 31-pair ablation was drawn from the same annotated candidate pool as the main set, with each meme evaluated under both context conditions. Its primary metric is Core4 (Table 8). Together, the evaluation covers 67 unique IDs and 98 meme-condition pages: 490 explanations, 1,470 explanation-level judgments, 7,350 scalar cells, and 294 holistic rankings converted into 2,940 pairwise outcomes. This dense design targets controlled within-meme model comparison rather than population-level estimation. Main-set score agreement is moderate (Krippendorf ordinal α ≈ 0.48).

## 6 Experiments

Unless noted, all five models are scored against final gold labels under the shared Task 1 manifests. Task 2 human ratings use the completed three-annotator blind evaluation protocol (Sect. 5); holistic rankings are reported as pairwise outcomes.

## 6.1 Task 1: Primary Order Recovery

Table 4 reports group-aware logical-order accuracy on the primary condition (strong, shufled, no context). The two italicized rows are weak references evaluated with the same group-aware parser and metric: Display-order follows the presented panel order, while Random samples one uniformly random permutation per item (seed = 42). Key findings: Open-source models remain weak on shufled order recovery: Qwen3.5-9B is best at 27.4%, while GLM-4.1V-9B-Thinking and InternVL3.5-8B reach 15.6% and 6.1%, respectively. Closed-source models are substantially stronger—Gemini 3.1 Pro reaches 75.2% and GPT-5.5 57.8%—but both still leave considerable room for improvement. Across both model families, canonical-display accuracy does not transfer reliably to shufled inputs (Table 6): the correct-versus-shufled gap persists, and Gemini’s shufled cells (73.0% / 71.3%) remain clearly below its correct-display cells (90.5% / 94.8%), confirming persistent order sensitivity.

Table 4. Task 1 primary order recovery (gold-strong, shufled, no context). Acc. denotes group-aware logical-order accuracy (%).
<table><tr><td>Model</td><td>n scored Acc. (%)</td></tr><tr><td>InternVL3.5-8B</td><td>863 6.1</td></tr><tr><td>Qwen3.5-9B</td><td>865 27.4</td></tr><tr><td>GLM-4.1V-9B-Thinking</td><td>865 15.6</td></tr><tr><td>GPT-5.5</td><td>868 57.8</td></tr><tr><td>Gemini 3.1 Pro</td><td>868 75.2</td></tr><tr><td>Display-order</td><td>868 0.0</td></tr><tr><td>Random (seed=42) 868</td><td>19.4</td></tr></table>

Model rows use parseable outputs; the smaller open-model counts reflect malformed or incomplete order outputs, including failures on longer inputs (especially beyond four panels).

## 6.2 Task 1: Structure Typing

Table 5 summarizes type prediction on all 1,214 memes. Open-source models remain in the mid-50% range, with the best accuracy at 56.1% and macro- $F _ { 1 }$ scores of 41.8–50.7. Closed-source models perform better: GPT-5.5 reaches 76.3% accuracy / 74.9 macro- $F _ { 1 }$ , while Gemini 3.1 Pro reaches 69.7% / 67.9. With parse rates above 99.7%, the bottleneck is semantic structure recognition rather than output validity, especially at the narrative–conversational and parallel– comparison boundaries (Sect. 7).

Table 5. Task 1 structure typing (N = 1214, canonical order, no context). Metrics in %.
<table><tr><td>Model</td><td></td><td>Acc. Macro-F1</td><td>W.-F1</td><td>Parse</td></tr><tr><td>InternVL3.5-8B</td><td>53.6</td><td>41.8</td><td>48.7</td><td>99.7</td></tr><tr><td>Qwen3.5-9B</td><td>56.1</td><td>47.4</td><td>51.2</td><td>100.0</td></tr><tr><td>GLM-4.1V-9B-Thinking</td><td>55.8</td><td>50.7</td><td>52.5</td><td>99.8</td></tr><tr><td>GPT-5.5</td><td>76.3</td><td>74.9</td><td>76.4</td><td>100.0</td></tr><tr><td>Gemini 3.1 Pro</td><td>69.7</td><td>67.9</td><td>67.0</td><td>100.0</td></tr></table>

## 6.3 Context Ablation (CAS)

Table 6 reports logical-order accuracy on the CAS 2 × 2 grid. Shufled displays remain substantially harder than correct displays, and comments do not close this gap for either open- or closed-source models. Closed-source models are more robust under shufling, with GPT-5.5 at approximately 51% and Gemini 3.1 Pro at 71–73%, but both remain order-sensitive. Comment efects are inconsistent:

InternVL improves from 4.8% to 6.8% under shufle, Qwen stays near 23%, and Gemini declines slightly with comments. Thus, post-context ofers limited, model-specific assistance but cannot substitute for visual sequence reasoning.

Table 6. CAS logical-order accuracy (%). Each cell uses comment-available order-pool memes (n ≈ 250–252 scored).
<table><tr><td rowspan="2">Model</td><td>Correct display Shuffled display</td><td></td><td></td><td></td></tr><tr><td>no ctx</td><td>+cmts</td><td>no ctx</td><td>+cmts</td></tr><tr><td>InternVL3.5-8B</td><td>92.8</td><td>85.6</td><td>4.8</td><td>6.8</td></tr><tr><td>Qwen3.5-9B</td><td>85.7</td><td>88.9</td><td>23.5</td><td>23.1</td></tr><tr><td>GLM-4.1V-9B-Thinking</td><td>97.2</td><td>96.0</td><td>10.4</td><td>13.5</td></tr><tr><td>GPT-5.5</td><td>92.9</td><td>91.7</td><td>51.2</td><td>51.2</td></tr><tr><td>Gemini 3.1 Pro</td><td>90.5</td><td>94.8</td><td>73.0</td><td>71.3</td></tr></table>

## 6.4 Dificulty by Length

On the primary condition, long strips are markedly harder than shorter ones: GPT-5.5 drops from 59.1% on main\_le6 to 16.0% on hard\_7plus, and Gemini 3.1 Pro from 76.7% to 24.0%; open models likewise remain near floor on hard items (InternVL 4.8%, Qwen 8.0%, GLM 8.3%). Across all scored Task 1 order requests, the panel\_4plus bucket is also relatively dificult (GPT-5.5 63.7%, Gemini 74.2%), trailing the corresponding 2-/3-panel overall rates.

## 6.5 Task 2: Explanation Generation

Table 7 reports five-dimension mean scores on the main set (64 unique contextpresent memes). As a secondary consistency check, each independently collected five-way ordering is expanded into pairwise wins and ties rather than reconstructed from summed Likert scores. Table 8 reports the paired Core4 ablation separately. These pairwise outcomes recover the broad tiers in the scalar ratings: Gemini 3.1 Pro and GPT-5.5 are preferred over the open models, while GLM is slightly favored over Qwen in their direct comparison. The gap is substantial (Gemini ≈2.7–2.8 vs. InternVL ≈1.9–2.2). Adding post context raises average Core4 only from 2.30 to 2.36 (∆= + 0.06); using a per-meme change of at least 0.1, context helps 13 pairs, hurts 10, and has a small or mixed efect on 8. Context is therefore conditionally useful rather than universally beneficial, and inter-annotator agreement is moderate (ordinal α ≈ 0.48).

Table 7. Task 2 main scalar evaluation (64 unique context-present memes). Mean of five Likert dimensions (1–3) per annotator. Higher is better; relative preferences are derived separately from holistic rankings.
<table><tr><td>Model</td><td>ann_A ann_B</td><td>ann C</td></tr><tr><td>InternVL3.5-8B</td><td>1.91</td><td>2.03 2.16</td></tr><tr><td>Qwen3.5-9B</td><td>2.00 2.08</td><td>2.33</td></tr><tr><td>GLM-4.1V-9B-Thinking</td><td>2.03 2.36</td><td>2.47</td></tr><tr><td>GPT-5.5</td><td>2.43 2.49</td><td>2.65</td></tr><tr><td>Gemini 3.1 Pro</td><td>2.78 2.71</td><td>2.79</td></tr></table>

Table 8. Task 2 context ablation (ABC combined; 31 matched pairs from the curated Task 2 evaluation pool; n=93 ratings per model–condition cell and n=465 for the overall aggregate). Core4 = mean of Visual, Panel, Humor, and Faithfulness (excludes Context).
<table><tr><td>Model</td><td>full_context image_only</td><td></td><td>Δ</td></tr><tr><td>InternVL3.5-8B</td><td>2.03</td><td>1.87</td><td>+0.16</td></tr><tr><td>Qwen3.5-9B</td><td>2.09</td><td>2.09</td><td>+0.00</td></tr><tr><td>GLM-4.1V-9B-Thinking</td><td>2.38</td><td>2.37</td><td>+0.01</td></tr><tr><td>GPT-5.5</td><td>2.52</td><td>2.49</td><td>+0.03</td></tr><tr><td>Gemini 3.1 Pro</td><td>2.77</td><td>2.69</td><td>+0.08</td></tr><tr><td>Overall</td><td>2.36</td><td>2.30</td><td>+0.06</td></tr></table>

## 7 Analysis and Discussion

## 7.1 Order Sensitivity

The dominant failure mode is order blindness under shufle. Open models often emit plausible left-to-right sequences that ignore causal, contrastive, or dialogue constraints after permutation. Closed-source models narrow but do not eliminate the gap (Table 6): GPT-5.5 remains near 51% under shufle, while Gemini 3.1 Pro remains well below its correct-display scores. Thus canonical accuracy can reflect exploitation of the presented order, whereas shufled evaluation tests whether panel recognition is integrated into reliable sequence reasoning.

## 7.2 Structure Confusions

Errors concentrate at semantic boundaries, especially narrative–conversational and parallel–comparison; the rarer progressive class is often absorbed into them. This mirrors annotation boundary cases and shows that LVLMs do not consistently encode structural distinctions among layouts.

## 7.3 Does Comment Context Help?

Comments are not a universal remedy: they help when restating a punchline but can mislead when ironic, referential, or unrelated to panel order. Visual sequence reasoning remains necessary.

## 7.4 Qualitative Error Modes

Figure 3 highlights three recurring failure modes: contrast swaps reverse the intended progression, conversational errors misorder reply turns, and punchlinefirst errors promote a salient final panel. Together, these cases show that order recovery requires discourse relations, not panel recognition alone.

![](images/8ffb9d1850f867ad358738d3704c43e390824ccbd7fedd523b1c0e2ef524b88b.jpg)  
Fig. 3. Illustrative Task 1 order-recovery errors under shufled display. Numbers are shufled display indices; each subfigure contrasts gold and observed orders.

## 7.5 Limitations

CMPM focuses on Chinese multi-panel social memes and may not cover other cultures or single-image templates. Existing LVLM benchmarks more often use single images or document pages, so multi-panel sequence settings remain less common. The study evaluates frozen checkpoints and proprietary APIs, making results sensitive to prompts, decoding settings, and future model updates. Task 2 agreement is moderate $( \alpha \approx 0 . 4 8 )$ , and its curated subset targets controlled comparison rather than population-level estimation. Release will remove identifying metadata and honor takedown requests.

## 8 Conclusion

We introduced CMPM, a 1,214-sample Chinese multi-panel meme benchmark for order-sensitive reasoning. Shufling reveals order blindness: Gemini 3.1 Pro reaches 75.2% on the primary condition, while open models remain lower. Structure typing is more reliable than boundary-sensitive ordering; comment gains are heterogeneous. Task 2 favors Gemini 3.1 Pro and GPT-5.5, with moderate agreement cautioning against a single score. Future work has three levels: expand data across languages, cultures, panel counts, and adversarial permutations; add panel-aware representations, cross-panel memory, and lightweight order-consistency adapters; and develop equivalence-aware, attention-grounded evaluation across model versions.

## References

1. Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966 (2023)

2. Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al.: MMStar: A vision-language benchmark for multimodal large language models. arXiv preprint arXiv:2403.20330 (2024)

3. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238 (2024)

4. Google DeepMind: Gemini 3.1 Pro model card. https://deepmind.google/ models/model-cards/gemini-3-1-pro/ (2026)

5. Hou, B., Lin, J., Zhang, C., Yin, D., Zhu, S., Hong, Q., Gao, M., Wang, J.: CHROMIC: Chronological reasoning across multi-panel comics. In: Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (2026)

6. Iyyer, M., Manjunatha, V., Guha, A., Vyas, Y., Boyd-Graber, J., Daumé III, H., Davis, L.: The amazing mysteries of the gutter: Drawing inferences between panels in comic book narratives. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing (2017)

7. Kiela, D., Firooz, H., Mohan, A., Goswami, V., Singh, A., Ringshia, P., Testuggine, D.: The hateful memes challenge: Detecting hate speech in multimodal memes. In: Advances in Neural Information Processing Systems (NeurIPS) (2020)

8. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Advances in Neural Information Processing Systems (NeurIPS) (2023)

9. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: MMBench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 (2024)

10. OpenAI: Introducing GPT-5.5. https://openai.com/index/introducing-gpt-5-5/ (2026)

11. OpenGVLab: InternVL3.5-8B model card. Hugging Face model card, https://huggingface.co/OpenGVLab/InternVL3\_5-8B (2026)

12. Qwen Team: Qwen3.5-9B model card. Hugging Face model card, https://huggingface.co/Qwen/Qwen3.5-9B (2026)

13. Sánchez Villegas, D., Ziegler, I., Elliott, D.: ImageChain: Advancing sequential image-to-text reasoning in multimodal large language models. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (2026)

14. Sharma, C., Bhageria, D., Scott, W., Pykl, S., Das, A., Chakraborty, T., Pulabaigari, V., Gambäck, B.: SemEval-2020 task 8: Memotion analysis—the visuolingual metaphor! In: Proceedings of SemEval (2020)

15. Suryawanshi, S., Chakravarthi, B.R., Arcan, M., Buitelaar, P.: Multimodal meme dataset (MultiOFF) for identifying ofensive content in image and text. In: Proceedings of TRAC (2020)

16. Vivoli, E., Bertini, M., Karatzas, D.: CoMix: A comprehensive benchmark for multi-task comic understanding. In: Advances in Neural Information Processing Systems (NeurIPS) (2024)

17. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-VL: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

18. Xie, Y., et al.: Are large language models chronically online surfers? a dataset for chinese internet meme explanation. arXiv preprint arXiv:2510.00567 (2025)

19. Z.ai: GLM-4.1V-9B-thinking model card. Hugging Face model card, https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking arXiv:2507.01006

(2026),

20. Zhao, Z., Zhang, S., Zhang, Y., Zhao, Y., Zhang, Y., Wang, Z., Wang, H., Zhao, Y., Liang, B., Zheng, Y., Li, B., Wong, K.F., Wu, X.: MemeReaCon: Probing contextual meme understanding in large vision-language models. In: Proceedings of EMNLP (2025). https://doi.org/10.18653/v1/2025.emnlp-main.176

21. Zhu, F., Liu, Z., Yao, N.X., Wu, H., Wang, W., Feng, F., Wang, C., Luan, H., Chua, T.S.: MMDocBench: Benchmarking large vision-language models for fine-grained visual document understanding and grounding. In: Proceedings of MMM. Lecture Notes in Computer Science, vol. 16412, pp. 74–88. Springer (2026). https://doi.org/10.1007/978-981-95-6950-2\_6