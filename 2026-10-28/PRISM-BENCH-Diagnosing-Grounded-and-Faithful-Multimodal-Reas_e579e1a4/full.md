# PRISM-BENCH: Diagnosing Grounded and Faithful Multimodal Reasoning Beyond Final-Answer Accuracy

Vansh Bhatia <sup>1</sup> <sup>♣</sup> Mehdi Rezagholizadeh <sup>1</sup> <sup>♣</sup> Parsa Ashrafi Fashi <sup>1</sup>

Igor Gilitschenski <sup>2</sup> Vikram Appia <sup>1</sup> Emad Barsoum <sup>1</sup>

<sup>1</sup>Advanced Micro Devices, Inc. <sup>2</sup>University of Toronto

♣ Corresponding authors: {vansh.bhatia, mehdi.rezagholizadeh}@amd.com

## Abstract

Final-answer accuracy reports whether a vision-language model was right, but not whether its answer was grounded in the image, whether its stated reasoning reflects that evidence, or what the answer cost to produce. Those are the properties that decide whether a system can be deployed. We present PRISM-BENCH, a benchmark of 1,000 expert-curated questions spanning five under-represented reasoning categories, in which every item carries an idealized reasoning trace written by one human annotator and validated by another. The traces support two diagnostics that accuracy cannot provide. A perception probe, in which the model answers from a description it wrote itself, attributes roughly 85% of frontier-model failures to misperception and places the bottleneck at grounding. Trace alignment predicts answer correctness for all 25 systems tested (macro AUROC 0.66) and survives length control, so it measures reasoning content and not verbosity. We further introduce CORE, an eficiency-aware metric that reorders 24.2% of model pairs relative to accuracy while remaining uncorrelated with response length. Across 25 open and closed systems, including GPT-5 and Claude Opus 4.6, the best accuracy is 66.0%. On the four categories with a human baseline the best system trails humans by 18.4 points, and recent frontier gains fall on recognition and not on spatial grounding.

Dataset: https://huggingface.co/datasets/vanshbhatia/PRISM-BENCH

## 1 Introduction

Vision-language models are moving from describing a scene to acting within it, in robotics, driver assistance, and clinical decision support. That shift changes what an evaluation has to report. Correctness remains necessary, but a deployed system must also be grounded, so that its answer follows from what the image contains and not from a linguistic prior; faithful, so that its stated reasoning reflects the evidence it actually used; and afordable, now that reasoning-heavy inference dominates serving cost.

Standard multimodal benchmarks report none of these properties, because they score only the final answer. A model that arrives at the right option through a hallucinated causal story is indistinguishable from one that reasons soundly, and a model that spends ten thousand tokens looks identical to one that spends a hundred. In deployment these are diferent systems. The first fails silently once the input distribution shifts; the second fails the latency budget.

We present PRISM-BENCH, a benchmark built so that these properties can be measured. It contains 1,000 expert-curated questions across five under-represented reasoning categories, and each item is paired with an idealized reasoning trace written by a human annotator: the minimal, logically complete chain a careful person would follow. Because that reference supplies a per-item ground truth for the reasoning process and not only for the outcome, it makes diagnostics available that an accuracy-only benchmark cannot support.

Figure 1 summarizes the three diagnostics this makes possible. Our contributions are:

(b) Trace alignment

(a) Grounding

![](images/b1024596457772d5c23c87c62d80f88a74f4e48519a81ef72d0e9ea1863e63ad.jpg)

![](images/15d53406561f472adcb32d64ab5e76922d981153e609665216905ce0ef8bcf2a.jpg)

![](images/9981d2d248016ae66b7047e0ba6ab263141cb3fa7200b068b7af0e40d648a767.jpg)  
Figure 1: Three deployment-relevant diagnostics that final-answer accuracy cannot provide. (a) Accuracyfalls when a model must answer from its own image description instead of the image (drop annotated in red); the white figure inside each bar is the share of that model’s baseline failures that the model itself, judging in a fresh context, attributes to perception instead of inference. (b) For all 25 evaluated systems, responses that are correct align more closely with the human reasoning trace than responses that are incorrect (macro AUROC 0.66, and 0.665 after regressing out response length). (c) CORE reorders models relative to accuracy; dashed arrows connect matched thinking/instruct pairs, which spend more tokens for less accuracy.

• The benchmark. 1,000 questions across five categories, each with a ground-truth answer, a human-authored reasoning trace, and multi-label reasoning tags, built by a four-stage pipeline in which no annotator validates an item they authored.

• A grounding diagnostic. A perception probe attributes roughly 85% of frontier failures to misperception, and accuracy falls by up to 11.5 points when a model must work from its own description of the image.

• A reasoning-content diagnostic. Reference-trace alignment separates correct from incorrect responses for all 25 systems tested, at macro AUROC 0.66 and 0.665 after length control.

• A cost-aware metric. CORE scores accuracy per unit of reasoning cost, reordering 24.2% of model pairs relative to accuracy while remaining uncorrelated with raw length (ρ = −0.05).

• Evaluation and self-audit. 25 open and closed systems, including GPT-5 and Claude Opus 4.6, against an expanded human baseline, with a reported audit of our protocol’s ranking stability, critic reliability, and contamination exposure.

## 2 Related Work

Multimodal reasoning benchmarks. A rich ecosystem probes visual mathematics [11, 24, 15], spatial and relational understanding [8, 5], abstraction [2, 10], and domain-specific settings [12, 13, 1, 7]. Broader suites such as MMMU-Pro [23] and MegaBench [4] reduce shortcut exploitation. These benchmarks largely score final answers, rarely release step-level reference reasoning, and do not report generation cost. As a result they cannot separate a grounded solution from a lucky one, which is the distinction that matters at deployment.

Reasoning and process supervision. Chain-of-thought prompting [26] and RL-style reasoning training [6, 14] have been extended to multimodal models via GRPO-style optimization [18, 19], self-reflection [20], and structured captioning [22]. Process reward models including VisualPRM [21] and DreamPRM [3] score intermediate steps, but current PRMs show unstable labelling and weak out-of-distribution generalization [17], a limitation we quantify on our own data in Section 5.5. Test-time compute scaling [16] makes cost-aware evaluation increasingly necessary, and verbose reasoning has been linked to hallucination [25].

![](images/bdbe219a096a61634cfc1f26b5ce166044760f3777f00defdd7cadfc410f05fd.jpg)  
Figure 2: The five reasoning categories, balanced at 200 items each. The categories mix functional reasoning patterns, such as multi-hop composition, with domain-specific grounding requirements, such as medical interpretation, because deployment demands both.

Table 1: PRISM-BENCH dataset statistics. The three answer-choice rows partition all 1,000 items; most are 4-way or wider multiple choice, and 11 items are open-ended or binary.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total questions</td><td>1,000</td></tr><tr><td>Categories</td><td>5 (200 each)</td></tr><tr><td>Average question length</td><td>24.7 words</td></tr><tr><td>Average trace length (structured / free-form)</td><td>70.1 / 43.8 words</td></tr><tr><td>Questions with 0–2 answer choices</td><td>11</td></tr><tr><td>Questions with 3–4 answer choices</td><td>245</td></tr><tr><td>Questions with 4+ answer choices</td><td>744</td></tr><tr><td>Annotation time per question</td><td>20–30 min</td></tr><tr><td>Image sources</td><td>RefCOCO, OpenStreetMap, MedXpertQA, handwriting volunteers, IAM</td></tr></table>

## 3 PRISM-BENCH

## 3.1 Categories

PRISM-BENCH contains 1,000 manually curated questions, 200 in each of five categories (Figure 2), chosen to probe what we call the Perception–Cognition Disconnect: cases where a model perceives visual elements yet fails to build a coherent world model. We deliberately exclude saturated categories such as chart and math QA, where memorized templates are exploitable. Table 1 gives the dataset statistics.

1. Visual Coreference / Occlusion (200): resolving references to partially occluded targets among distractors, requiring object identity and spatial logic under incomplete evidence.

2. Handwriting / OCR Reasoning (200): recognizing handwritten text and reasoning over it, where one misread character cascades downstream.

3. Navigation / Map-based Reasoning (200): interpreting schematic maps and composing multi-step spatial instructions.

4. Multi-hop Reasoning (200): chaining visual and textual clues across image regions.

5. Scientific / Medical Reasoning (200): interpreting medical visual cues with domain causal knowledge, where models often perceive correctly but hallucinate causal explanations.

Each item carries optional answer choices, a ground-truth answer, a human-authored idealized reasoning trace, and multi-label reasoning tags (visual, spatial, textual, commonsense). The trace is recorded in two human-written forms: the annotator’s original free-form chain (mean 43.8 words) and a structured step-by-step rendering of the same chain (mean 70.1 words). CORE takes its length reference from the structured form, and trace alignment compares against the original.

## 3.2 Construction

Images come from RefCOCO [9] (coreference, multi-hop), schematic layouts derived from OpenStreetMap (navigation), handwriting scans and IAM, and MedXpertQA [27] (medical). A four-stage pipeline enforces separation of duties, so no annotator both authors and validates the same item: (1) experts design questions that decouple surface cues from the reasoning target; (2) annotators author idealized minimal traces, manually audited and edited by experts; (3) annotators assign multi-label tags; (4) a separate annotator validates answer correctness, trace validity, and clarity, with failing items revised or removed. Each item requires 20–30 minutes of total annotation efort.

Question, option, answer, and trace text is newly authored for 807 of 1,000 items (80.7%); only source images are reused. The 193 medical items reuse an expert-validated image, question, and answer, with the trace authored to that verified answer.

## 4 Cost-Aware Metrics

Two models with equal accuracy can difer by an order of magnitude in tokens. CORE scores correctness per unit reasoning cost, using the human trace length as the per-item reference:

$$
\begin{array} { r } { C O R E ( a , 1 ) = \frac { a } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \log { l _ { i } } } { \log { L _ { i } ^ { \mathrm { g t } } } } } , } \end{array}\tag{1}
$$

where a is accuracy, $l _ { i }$ the generated length, and $L _ { i } ^ { \mathrm { g t } }$ the human trace length. When $l _ { i } = L _ { i } ^ { \mathrm { g t } }$ the denominator is 1 and CORE reduces to accuracy. We normalize in log space because length distributions are heavy-tailed, so that a single verbose response cannot dominate the average. The reference serves as an eficiency floor and is not a claim that the human path is the only valid one: the log ratio tolerates alternative correct derivations of moderately diferent length.

The critic-augmented variant incorporates step-level process reward scores $v _ { i } \in [ 0 , 1 ]$ to discount correct-but-ungrounded answers:

$$
C O R E - P R M ( \mathbf { a } , \mathbf { v } , \mathbf { l } ) = \frac { 1 } { 2 N } \sum _ { i = 1 } ^ { N } \frac { ( a _ { i } + v _ { i } ) \log { \cal L } _ { i } ^ { \mathrm { g t } } } { \log { l _ { i } } } .\tag{2}
$$

CORE is critic-free and is our primary metric; CORE-PRM is a critic-agnostic template whose validity is bounded by the judge supplied to it (Section 5.5).

## 5 Experiments

Models generate an answer plus a step-by-step rationale, via chain-of-thought prompting or native thinking mode Open-weight models use greedy decoding, giving zero run-to-run variance, and all models share a fixed prompt template and answer-extraction cascade.

## 5.1 Frontier Models Remain Far Below Humans

No model exceeds 66% overall, against a human average of 83.9% (Table 2). The two aggregates cover diferent category sets, since the human study excludes Scientific/Medical, so the like-for-like comparison is over the four categories both measure: there the best system reaches 65.5%, a gap of 18.4 points. The shape of that gap matters more than its size. The newest frontier systems post the strongest OCR scores, above Gemini’s, yet their navigation scores are the weakest of the four, near 47% where humans reach 86%. Recent progress has concentrated in recognition, while spatial grounding, the capability most relevant to embodied and autonomous settings, has not moved.

Table 2: Accuracy (%) of all 25 evaluated systems across the five PRISM-BENCH categories, against the human baseline (5 annotators × 50 items per category). No system exceeds 66% overall, or 84% in any single category. The newest frontier systems lead on OCR yet remain near 47% on navigation, where humans reach 86%: recent gains concentrate on recognition rather than spatial grounding.
<table><tr><td>Model</td><td>Size</td><td>Overall</td><td>Coref.</td><td>OCR</td><td>Nav.</td><td>Multi-hop</td><td>Sci./Med.</td></tr><tr><td>Closed-source</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 2.5 Pro</td><td></td><td>66.0</td><td>57.5</td><td>77.0</td><td>58.2</td><td>69.2</td><td>68.5</td></tr><tr><td>GPT-5</td><td></td><td>63.6</td><td>47.0</td><td>82.0</td><td>47.0</td><td>73.6</td><td>68.0</td></tr><tr><td>Claude Opus 4.6</td><td>一</td><td>62.4</td><td>51.5</td><td>83.5</td><td>47.8</td><td>69.7</td><td>59.3</td></tr><tr><td>OpenAI o3</td><td></td><td>57.2</td><td>39.5</td><td>75.5</td><td>46.8</td><td>65.2</td><td>59.5</td></tr><tr><td>Claude Sonnet 4</td><td></td><td>39.6</td><td>39.0</td><td>46.5</td><td>39.8</td><td>38.3</td><td>34.5</td></tr><tr><td>Open-weight (instruct)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL-30B-A3B-Inst.</td><td>30B</td><td>42.5</td><td>41.0</td><td>52.5</td><td>35.3</td><td>48.3</td><td>35.5</td></tr><tr><td>Qwen3-VL-8B-Inst.</td><td>8B</td><td>40.0</td><td>39.5</td><td>50.5</td><td>42.3</td><td>40.3</td><td>27.5</td></tr><tr><td>Gemma-3-12B-it</td><td>12B</td><td>34.6</td><td>39.0</td><td>40.5</td><td>27.4</td><td>38.3</td><td>28.0</td></tr><tr><td>InternVL3.5-8B</td><td>8B</td><td>30.2</td><td>29.5</td><td>36.5</td><td>27.4</td><td>30.9</td><td>27.0</td></tr><tr><td>Gemma-3-4B-it</td><td>4B</td><td>29.0</td><td>36.0</td><td>30.5</td><td>23.9</td><td>32.8</td><td>22.0</td></tr><tr><td>Qwen2.5-VL-7B-Inst.</td><td>7B</td><td>27.4</td><td>30.0</td><td>31.5</td><td>32.8</td><td>22.9</td><td>20.0</td></tr><tr><td>InternVL3-8B-Inst.</td><td>8B</td><td>26.7</td><td>30.5</td><td>25.0</td><td>26.9</td><td>30.9</td><td>20.5</td></tr><tr><td>Qwen2.5-VL-3B-Inst.</td><td>3B</td><td>22.1</td><td>24.5</td><td>19.0</td><td>28.9</td><td>19.4</td><td>18.5</td></tr><tr><td>InternVL3-2B-Inst.</td><td>2B</td><td>20.7</td><td>31.0</td><td>16.0</td><td>22.4</td><td>23.9</td><td>10.0</td></tr><tr><td>Molmo-7B-D-0924</td><td>7B</td><td>20.1</td><td>25.5</td><td>21.0</td><td>20.9</td><td>19.4</td><td>14.0</td></tr><tr><td>Open-weight (thinking / RL-trained)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Skywork-R1V3-38B</td><td>38B</td><td>38.6</td><td>42.5</td><td>42.5</td><td>32.3</td><td>44.3</td><td>31.5</td></tr><tr><td>Qwen3-VL-30B-A3B-Think.</td><td>30B</td><td>36.8</td><td>34.0</td><td>57.5</td><td>36.8</td><td>38.8</td><td>17.0</td></tr><tr><td>Skywork-R1V2-38B</td><td>38B</td><td>36.7</td><td>40.0</td><td>32.5</td><td>30.9</td><td>48.8</td><td>31.5</td></tr><tr><td>Ovis2.5-9B</td><td>9B</td><td>36.1</td><td>36.5</td><td>40.5</td><td>37.8</td><td>42.8</td><td>22.5</td></tr><tr><td>Qwen3-VL-8B-Think.</td><td>8B</td><td>36.0</td><td>32.0</td><td>54.0</td><td>33.3</td><td>33.3</td><td>27.5</td></tr><tr><td>VL-Rethinker</td><td>7B</td><td>29.1</td><td>32.0</td><td>35.0</td><td>31.3</td><td>30.4</td><td>16.5</td></tr><tr><td>Kimi-VL-A3B-Think.</td><td>16B</td><td>29.0</td><td>34.5</td><td>33.5</td><td>32.8</td><td>26.9</td><td>17.5</td></tr><tr><td>Ovis2.5-2B</td><td>2B</td><td>28.4</td><td>32.5</td><td>30.5</td><td>32.3</td><td>29.4</td><td>17.0</td></tr><tr><td>Visionary-R1</td><td>7B</td><td>26.2</td><td>30.5</td><td>23.0</td><td>31.8</td><td>24.4</td><td>21.5</td></tr><tr><td>GLM-4.1V-9B-Think.</td><td>9B</td><td>23.8</td><td>29.0</td><td>18.5</td><td>24.4</td><td>27.4</td><td>19.5</td></tr><tr><td>Human baseline</td><td>一</td><td>83.9</td><td>84.0</td><td>91.2</td><td>86.0</td><td>74.4</td><td>n/a</td></tr></table>

Figure 3 makes this explicit. Across all 25 systems, none approaches the human baseline, and the remaining deficit is almost entirely spatial. On multi-hop, where the reasoning is largely textual once the scene has been read, the best system comes within 0.8 points of humans; on coreference and navigation it trails by 26.5 and 27.8 points. A system can look competitive on an aggregate multimodal leaderboard and still be unusable for the embodied tasks that motivate deployment. Instruct variants also match or exceed their thinking counterparts at matched scale (40.0% against 36.0% for Qwen3-VL-8B, and 42.5% against 36.8% at 30B).

Two further patterns are stable across Table 2. Scale is not decisive: Ovis2.5-9B outperforms several 30B+ systems on navigation, so category specialization matters more than parameter count. And navigation and scientific/medical remain the hardest categories for nearly every system, while OCR is the most tractable.

## 5.2 Grounding: Most Failures Are Perceptual

To locate where models break, we run a two-stage probe on the two categories that depend most on grounding. The model first describes the image; it is shown the question but told not to answer it, so the description is directed at what the item needs. It then answers from that description alone, with the visual input removed. Finally, in a fresh text-only context, the same model labels each baseline failure as either perceptual or inferential (Section A).

Two independent signals converge (Table 3, plotted in Figure 1a). The judge attributes about 85% of failures to perception. Independently of any judge, accuracy drops when a model must work from its own description, by 3.7 points for GPT-5 on navigation and by 11.5 for Opus on coreference, which means the descriptions omit information the model needed. Were the failures inferential, re-reading a faithful self-description would cost nothing.

(b) The gap is spatial, not textual  
(a) No system approaches the human baseline  
![](images/76d4cac07447f90c4b7ed92c51d5e56217cdf1aa13046f7df933903d3a2dc583.jpg)

![](images/e34e2fde8778173c7c2079eef95d7cbeccbd6478e0ad5446f40e55e3a5a520ba.jpg)

Figure 3: (a) Overall accuracy for all 25 evaluated systems, shaded by family, against the 83.9% human baseline (red). Restricted to the four categories where both are measured, the best system trails humans by 18.4 points, and the open/closed gap is larger than any diference within the open families. (b) Best system of any kind versus the human baseline (red tick) per category. The deficit is concentrated in the two spatially grounded categories and nearly vanishes on multi-hop, where the reasoning is largely textual once the scene is read.  
Table 3: Perception-vs-reasoning probe on navigation and coreference. Accuracyfalls when models answer from their own descriptions, and the large majority of failures are attributed to perception.
<table><tr><td>Model</td><td>Category</td><td>Baseline acc.</td><td>From own description</td><td>% failures perceptual</td></tr><tr><td>GPT-5</td><td>Navigation</td><td>47.0</td><td>43.3</td><td>88</td></tr><tr><td>GPT-5</td><td>Coreference</td><td>47.0</td><td>41.5</td><td>85</td></tr><tr><td>Claude Opus 4.6</td><td>Navigation</td><td>47.8</td><td>38.8</td><td>90</td></tr><tr><td>Claude Opus 4.6</td><td>Coreference</td><td>51.5</td><td>40.0</td><td>79</td></tr><tr><td>Combined</td><td>Nav. + Coref.</td><td>~48-50</td><td>~39-42</td><td>~85</td></tr></table>

The practical consequence is that adding reasoning tokens to a model that has already mis-seen the scene cannot recover the answer. Efort is better spent on visual grounding than on longer chains, which is consistent with our finding that thinking variants underperform instruct variants at matched scale.

## 5.3 Reference-Trace Alignment Predicts Correctness

Does a model’s stated reasoning track the evidence, or is it a narrative attached to a guess? We segment each gold trace and model response into steps, embed them, and for each gold step take the maximum cosine similarity to any model step, yielding a semantic overlap score. This uses the traces for content, not length.

For all 25 systems, correct responses align more closely with the human trace than incorrect ones (Table 4; Figure 1b shows the full per-model spread). The macro AUROC of 0.66 is essentially unchanged at 0.665 once response length is regressed out, and all 25 remain above chance, so the efect belongs to reasoning content and not to verbosity. The result is an association: responses that cover the reference reasoning are more often correct, which does not by itself show that the stated reasoning is what produced the answer. Alignment is highest for OCR and lowest for multi-hop, which matches the intuition that compositional reasoning admits more divergent valid routes.

Table 4: Trace alignment predicts answer correctness. AUROC is computed per model over ∼1,000 items; the length-controlled column regresses out response length. All 25 systems exceed chance in both columns.
<table><tr><td>Model</td><td>AUROC (align → correct)</td><td>AUROC (length-controlled)</td></tr><tr><td>OpenAI o3</td><td>0.71</td><td>0.71</td></tr><tr><td>GPT-5</td><td>0.69</td><td>0.69</td></tr><tr><td>Claude Opus 4.6</td><td>0.68</td><td>0.67</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.67</td><td>0.68</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>0.67</td><td>0.68</td></tr><tr><td>Skywork-R1V3-38B</td><td>0.63</td><td>0.64</td></tr><tr><td>Macro-avg (25 systems)</td><td>0.66</td><td>0.665</td></tr></table>

Table 5: Cost-aware metrics for the 20 open-weight models with exact token counts. “Norm. len.” is mean response length divided by the human reference length. Ranks are computed within this table; 46 of the 190 model pairs (24.2%) are ordered diferently by accuracy and CORE. CORE is unavailable for closed systems that hide thinking tokens.
<table><tr><td>Model</td><td>Size</td><td>Acc. (%)</td><td>Norm. len.</td><td>CORE</td><td>VisualPRM v</td><td>CORE-PRM</td><td>Acc. rank</td><td>CORE rank</td></tr><tr><td>Gemma-3-4B-it</td><td>4B</td><td>29.0</td><td>1.04</td><td>28.01</td><td>0.64</td><td>45.48</td><td>11</td><td>5</td></tr><tr><td>Gemma-3-12B-it</td><td>12B</td><td>34.6</td><td>1.06</td><td>32.76</td><td>0.69</td><td>49.58</td><td>8</td><td>2</td></tr><tr><td>GLM-4.1V-9B-Think.</td><td>9B</td><td>23.8</td><td>1.26</td><td>18.93</td><td>0.26</td><td>20.05</td><td>17</td><td>20</td></tr><tr><td>InternVL3-2B-Inst.</td><td>2B</td><td>20.7</td><td>1.04</td><td>19.83</td><td>0.57</td><td>37.60</td><td>19</td><td>18</td></tr><tr><td>InternVL3-8B-Inst.</td><td>8B</td><td>26.7</td><td>1.06</td><td>25.18</td><td>0.65</td><td>43.80</td><td>15</td><td>12</td></tr><tr><td>InternVL3.5-8B</td><td>8B</td><td>30.2</td><td>1.05</td><td>28.76</td><td>0.66</td><td>45.90</td><td>9</td><td>4</td></tr><tr><td>Kimi-VL-A3B-Think.</td><td>16B</td><td>29.0</td><td>1.34</td><td>21.61</td><td>0.70</td><td>37.76</td><td>12</td><td>17</td></tr><tr><td>Molmo-7B-D-0924</td><td>7B</td><td>20.1</td><td>1.04</td><td>19.38</td><td>0.56</td><td>36.91</td><td>20</td><td>19</td></tr><tr><td>Ovis2.5-2B</td><td>2B</td><td>28.4</td><td>1.26</td><td>22.62</td><td>0.55</td><td>34.33</td><td>13</td><td>16</td></tr><tr><td>Ovis2.5-9B</td><td>9B</td><td>36.1</td><td>1.31</td><td>27.61</td><td>0.59</td><td>37.27</td><td>6</td><td>6</td></tr><tr><td>Qwen2.5-VL-3B-Inst.</td><td>3B</td><td>22.1</td><td>0.97</td><td>22.85</td><td>0.55</td><td>40.68</td><td>18</td><td>15</td></tr><tr><td>Qwen2.5-VL-7B-Inst.</td><td>7B</td><td>27.4</td><td>1.04</td><td>26.32</td><td>0.62</td><td>43.11</td><td>14</td><td>10</td></tr><tr><td>Qwen3-VL-8B-Think.</td><td>8B</td><td>36.0</td><td>1.51</td><td>23.92</td><td>0.70</td><td>36.40</td><td>7</td><td>14</td></tr><tr><td>Qwen3-VL-8B-Inst.</td><td>8B</td><td>40.0</td><td>1.25</td><td>31.92</td><td>0.72</td><td>46.26</td><td>2</td><td>3</td></tr><tr><td>Qwen3-VL-30B-A3B-Think.</td><td>30B</td><td>36.8</td><td>1.44</td><td>25.56</td><td>0.72</td><td>38.52</td><td>4</td><td>11</td></tr><tr><td>Qwen3-VL-30B-A3B-Inst.</td><td>30B</td><td>42.5</td><td>1.20</td><td>35.50</td><td>0.67</td><td>46.95</td><td>1</td><td>1</td></tr><tr><td>Skywork-R1V2-38B</td><td>38B</td><td>36.7</td><td>1.51</td><td>24.37</td><td>0.69</td><td>35.84</td><td>5</td><td>13</td></tr><tr><td>Skywork-R1V3-38B</td><td>38B</td><td>38.6</td><td>1.44</td><td>26.89</td><td>0.71</td><td>38.58</td><td>3</td><td>8</td></tr><tr><td>Visionary-R1</td><td>7B</td><td>26.2</td><td>0.99</td><td>26.52</td><td>0.56</td><td>42.15</td><td>16</td><td>9</td></tr><tr><td>VL-Rethinker</td><td>7B</td><td>29.1</td><td>1.06</td><td>27.54</td><td>0.63</td><td>44.00</td><td>10</td><td>7</td></tr></table>

This gives a critic-free signal of reasoning quality that requires no process reward model, and it doubles as a validation of the reference traces themselves: a single idiosyncratic annotator path could not predict correctness across 25 heterogeneous systems.

Toward hallucination detection at deployment time. Alignment is computed from the response alone against a stored reference, so it can be evaluated without knowing the answer, which is the situation a deployed monitor is in. A response whose steps diverge from the reference is measurably more likely to be wrong, making alignment usable as a cheap abstention or escalation signal. The signal is modest, and its limits are worth stating. An AUROC of 0.66 supports flagging low-confidence outputs for review, not verifying individual items, and the measure needs a reference trace, so it applies to curated evaluation sets and not to open-world inputs. It nonetheless ofers something a process reward model does not. In Section 5.5 we show that our of-the-shelf critic is close to non-discriminative on this data, whereas trace alignment separates correct from incorrect responses for every model we tested.

## 5.4 Deployment Cost: Longer Reasoning Does Not Pay

Table 5 gives the cost-aware view for the 20 open-weight models where we can count generated tokens exactly. Closed systems are excluded because their thinking tokens are not exposed, so their reasoning cost cannot be measured; this is a limitation of provider APIs, not of the metric. The table shows the deployment picture accuracy hides. At matched scale, Qwen3-VL-8B-Thinking generates 2.5× as many tokens as its instruct counterpart (1,812 versus 737) for lower accuracy, and CORE separates them by 8 points; the same holds at 30B. Gemma-3-12B-it, ranked 8th of 20 by accuracy, ranks 2nd by CORE on the strength of disciplined reasoning.

Table 6: Reflective-token analysis for 12 representative models. “Reflective tokens” counts hesitation and backtracking phrases (“Wait”, “Actually”, “Let me reconsider”) per response. More self-correction does not yield higher accuracy: Qwen3-VL-8B-Thinking averages 17.0 reflective phrases and 1,812 tokens to reach 36.0%, while its instruct counterpart averages 2.6 phrases and 737 tokens to reach 40.0%.
<table><tr><td>Model</td><td>Acc. (%)</td><td>Avg. tokens</td><td>Reflective tokens</td></tr><tr><td>Gemma-3-4B-it</td><td>29.0</td><td>195.5</td><td>0.1</td></tr><tr><td>Gemma-3-12B-it</td><td>34.6</td><td>198.4</td><td>0.1</td></tr><tr><td>Qwen3-VL-8B-Inst.</td><td>40.0</td><td>736.7</td><td>2.6</td></tr><tr><td>Qwen3-VL-8B-Think.</td><td>36.0</td><td>1811.7</td><td>17.0</td></tr><tr><td>Qwen3-VL-30B-A3B-Inst.</td><td>42.5</td><td>581.5</td><td>1.6</td></tr><tr><td>Qwen3-VL-30B-A3B-Think.</td><td>36.8</td><td>1253.2</td><td>12.0</td></tr><tr><td>Ovis2.5-2B</td><td>28.4</td><td>557.0</td><td>9.2</td></tr><tr><td>Ovis2.5-9B</td><td>36.1</td><td>658.6</td><td>11.9</td></tr><tr><td>Skywork-R1V2-38B</td><td>36.7</td><td>1380.9</td><td>9.8</td></tr><tr><td>Skywork-R1V3-38B</td><td>38.6</td><td>928.7</td><td>7.9</td></tr><tr><td>Kimi-VL-A3B-Think.</td><td>29.0</td><td>715.0</td><td>8.2</td></tr><tr><td>GLM-4.1V-9B-Think.</td><td>23.8</td><td>470.7</td><td>12.1</td></tr></table>

These reorderings are not a repackaging of length (Figure 1c). Across the 20 open-weight models with exact token counts, the Spearman correlation between CORE and normalized length is $\rho = - 0 . 0 5 0 \ : ( p = 0 . 8 3 4 )$ , so CORE is not an inverse-length score. Against accuracy it is $\rho = + 0 . 6 7 4 \ : ( p = 0 . 0 0 1 )$ , high enough that correctness dominates but well short of redundancy. Accuracy and normalized length are themselves correlated $( \rho = + 0 . 6 5 7 , p = 0 . 0 0 2 )$ , but this is a confound rather than a mechanism: more capable families happen to generate longer outputs, while within a matched-scale family the thinking variant spends 2.2–2.5× as many tokens for equal or lower accuracy. CORE-PRM tracks accuracy only weakly $( \rho = + 0 . 3 1 3 , p = 0 . 1 7 9 )$ . In all, 24.2% of model pairs (46 of 190) are ranked discordantly by the two measures. The mechanism is the per-item normalization in Equation (1): verbosity on an item a human solved briefly is penalized more heavily than the same verbosity on a deep item, and any aggregate length statistic discards that distinction.

Reflection tracks verbosity, not quality. Table 6 counts hesitation and backtracking phrases per response for 12 representative models, and reinforces the cost argument from a diferent direction. Gemma-3 models reach 29.0–34.6% with roughly 200 tokens and essentially no reflective phrases, while GLM-4.1V-9B-Thinking averages one self-correction every 39 tokens and posts the lowest accuracy in the table. For a deployed system this matters because reflective loop are pure cost: they lengthen the response, raise latency, and do not buy correctness.

## 5.5 Reliability of Our Own Protocol

Deployment claims should state their own uncertainty, so we report two audits of the evaluation itself.

Ranking stability. We evaluated three frontier systems under four benign prompt and decoding variants on a 150-item stratified subset, giving 1,799 runs with no errors. Each model’s accuracy is stable, deviating at most 4.8 points from its own mean, but the top-1 ranking flips. GPT-5 and Gemini sit within about two points of each other and exchange first place depending on the prompt, while Claude is last under every condition. Adjacent leaderboard positions among closely matched systems are therefore not identified by accuracy alone, which is a further reason to report grounding, faithfulness, and cost beside it.

Critic reliability. We tested whether VisualPRM-8B, our default critic, separates correct from incorrect reasoning on 200 balanced traces drawn from 5 categories and 8 models. It does so only weakly, with mean v of 0.608 against 0.596, and the gap is negative in multi-hop and coreference/occlusion. CORE-PRM should therefore be read as a critic-agnostic template and not as a settled score. This is why the critic-free CORE is our primary metric and carries every headline conclusion above. A stronger multimodal critic would improve CORE-PRM without any change to the benchmark.

Contamination. Because the source images are public, we ran a behavioral check: if exposure leaked answers, the most-reused sources would be the easiest. The opposite holds. GPT-5 and Opus 4.6 score 60.6 and 61.3 on COCO-RefCOCO images and 46.8 and 47.8 on OpenStreetMap schematics, making the two most public sources the two hardest, while handwriting scans, the least likely to have been memorized, are easiest at 82.1 and 83.6.

## 6 Failure Modes

Each category produces a distinct failure signature. In visual coreference, models over-attend to the most salient object and mishandle relative spatial phrases. In handwriting and OCR, a single misrecognized character cascades through the downstream arithmetic and corrupts an otherwise sound chain. In navigation, models identify landmarks correctly but fail at path composition, producing of-by-one directional errors and confusing left with right at intersections. In multi-hop, the first hop usually succeeds while the second introduces fabricated constraints. In the medical category, models frequently perceive the visual feature correctly and then hallucinate a causal attribution. That category has the highest rate of confidently wrong explanations and is the clearest safety concern for high-stakes deployment.

## 7 Conclusion

PRISM-BENCH evaluates multimodal reasoning along the axes that determine whether a system can be deployed: grounding, faithfulness, and cost. Pairing every question with a human-authored reasoning trace makes two diagnostics possible that accuracy cannot support. A perception probe attributes most frontier failures to grounding instead of inference, and a length-controlled alignment measure predicts correctness across 25 systems. Read together with a cost-aware metric that reorders a quarter of all model pairs, these results show that frontier systems remain far below human performance, that recent gains have concentrated in recognition while spatial grounding has stalled, and that additional reasoning tokens do not repair a scene the model has already mis-seen. We release the benchmark, traces, tags, and evaluation code at https://huggingface.co/datasets/vanshbhatia/PRISM-BENCH.

## Acknowledgments and Disclosure of Funding

This work was carried out at AMD. No external funding was received.

## References

[1] Jing Bi, Junjia Guo, Susan Liang, Guangyu Sun, Luchuan Song, Yunlong Tang, Jinxi He, Jiarui Wu, Ali Vosoughi, Chen Chen, and Chenliang Xu. Verify: A benchmark of visual explanation and reasoning for investigating multimodal reasoning fidelity, 2025.

[2] Huanqia Cai, Yijun Yang, and Winston Hu. Mm-iq: Benchmarking human-like abstraction and reasoning in multimodal models. arXiv preprint arXiv:2502.00698, 2025.

[3] Qi Cao and Pengtao Xie. Dreamprm-1.5: Unlocking the potential of each instance for multimodal process reward model training, 2025. URL https://arxiv.org/abs/2509.05542.

[4] Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, Dongfu Jiang, Xuan He, Yuan Liu, Hexiang Hu, Xiang Yue, and Wenhu Chen. Mega-bench: Scaling multimodal evaluation to over 500 real-world tasks. 2025.

[5] Erik Daxberger, Nina Wenzel\*, David Grifiths\*, Haiming Gang, Justin Lazarow, Gefen Kohavi, Kai Kang, Marcin Eichner, Yinfei Yang, Afshin Dehghan, and Peter Grasch. Mm-spatial: Exploring 3d spatial understanding in multimodal llms. In ICCV, 2025. URL https://arxiv.org/abs/2503.13111.

[6] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

[7] Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.

[8] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. arXiv preprint arXiv:2506.03135, 2025.

[9] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Alessandro Moschitti, Bo Pang, and Walter Daelemans, editors, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1086. URL https://aclanthology.org/D14-1086.

[10] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Jian Yang, Siwei Wu, Xingwei Qu, Jinjie Shi, Xinyue Zhang, Zhenzhu Yang, Xiangzhou Wang, Zhaoxiang Zhang, Zachary Liu, Emmanouil Benetos, Wenhao Huang, and Chenghua Lin. Omnibench: Towards the future of universal omni-language models, 2024. URL https://arxiv.org/abs/2409.15272.

[11] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

[12] Junyu Luo, Zhizhuo Kou, Liming Yang, Xiao Luo, Jinsheng Huang, Zhiping Xiao, Jingshu Peng, Chengzhong Liu, Jiaming Ji, Xuanzhe Liu, et al. Finmme: Benchmark dataset for financial multi-modal reasoning evaluation. arXiv preprint arXiv:2505.24714, 2025.

[13] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

[14] Niklas Muennighof, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025. URL https: //arxiv.org/abs/2501.19393.

[15] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.

[16] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more efective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

[17] Linzhuang Sun, Hao Liang, Jingxuan Wei, Bihui Yu, Tianpeng Li, Fan Yang, Zenan Zhou, and Wentao Zhang. Mm-verify: Enhancing multimodal reasoning with chain-of-thought verification, 2025. URL https: //arxiv.org/abs/2502.13383.

[18] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Xiansheng Chen, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning of vision language models, 2025. URL https://arxiv.org/abs/2503.20752.

[19] Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, Chaofan Tao, Yangfan He, Mi Zhang, and Shen Yan. Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning, 2025. URL https://arxiv.org/abs/2506.01713.

[20] Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. Vl-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning, 2025. URL https://arxiv.org/abs/ 2504.08837.

[21] Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, and Wenhai Wang. Visualprm: An efective process reward model for multimodal reasoning, 2025. URL https://arxiv.org/abs/2503.10291.

[22] Jiaer Xia, Yuhang Zang, Peng Gao, Sharon Li, and Kaiyang Zhou. Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning, 2025. URL https://arxiv.org/abs/2505.14677.

[23] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024.

[24] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024.

[25] Yusen Zhang, Sarkar Snigdha Sarathi Das, and Rui Zhang. Verbosity ̸= veracity: Demystify verbosity compensation behavior of large language models, 2024. URL https://arxiv.org/abs/2411.07858.

[26] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models, 2024. URL https://arxiv.org/abs/2302.00923.

[27] Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. Medxpertqa: Benchmarking expert-level medical reasoning and understanding, 2025. URL https://arxiv.org/abs/2501.18362.

## A Perception–Reasoning Probe Protocol

The probe in Section 5.2 runs in two stages on the two most grounding-dependent categories, navigation and coreference.

1. Baseline. The model answers the question with the image present, under the standard prompt and answer-extraction cascade used everywhere else in the paper.

2. Describe. The model is shown the same image together with the question and asked for an exhaustive description of everything in the image relevant to it: objects and their absolute and relative positions, visible text, labels and street names, colors, orientations, and counts. It is explicitly instructed not to answer. Because the description is conditioned on the question, it is directed at what the item needs rather than being a generic caption.

3. Answer from description. A fresh context receives the question and the model’s own description, with the image removed, and must answer from text alone.

4. Failure attribution. For each item the model failed at baseline, the same model, in a fresh text-only context, is shown the question and options, the gold answer, the gold free-form reasoning trace, and its own description. It labels the primary cause of failure as perceptual (the description omitted or misstated a visual fact needed to answer) or inferential (the description contained the needed facts but the answer was still wrong).

The two stages are deliberately redundant, because each has a weakness the other does not. The attribution step depends on a judge, so it inherits whatever bias that judge has. The accuracy comparison between stages 1 and 3 depends on no judge at all: if a model’s own description preserved the information it needed, re-reading that description could not reduce accuracy. Both point the same way, which is why we report the conclusion as approximately 85% and not as a single exact figure.

Scope. The probe covers two categories and two frontier systems, chosen because they are the settings where grounding is most clearly the binding constraint. We do not claim the 85% figure transfers to OCR or medical items, where the failure structure difers; the handwriting category in particular has a documented cascade pattern in which perception and inference are entangled by construction (Section E).

## B Trace Alignment Protocol

Each gold trace and each model response is segmented into steps by sentence and enumeration boundaries. Steps are embedded with BAAI/bge-small-en-v1.5 using normalized embeddings, and for every gold step we take the maximum cosine similarity to any step of the model response. The item score is the mean of those per-step maxima, so the measure asks whether the model covered the reference reasoning, not whether it matched it verbatim or matched its length. A response that reaches the same conclusion by a diferent valid route scores well as long as it touches the same intermediate facts.

For each model we then compute AUROC of this score against binary answer correctness over the ∼1,000 items. The length-controlled variant regresses response length out of the alignment score first and recomputes AUROC on the residual, which is what distinguishes a content efect from a verbosity artifact.

Table 7 reports per-model means and gaps for all 25 systems. The gap is positive in 25 of 25 cases, ranging from +0.0177 (GLM-4.1V-9B-Thinking) to +0.0534 (Qwen2.5-VL-7B-Instruct). Absolute similarities are high for every model and every outcome, since any on-topic response about the same image shares vocabulary with the reference. The gap, not the absolute value, therefore carries the signal.

## C Ranking Stability Protocol

The stability check in Section 5.5 evaluates three frontier systems under four conditions on a 150-item stratified subset (30 items per category), for 1,799 valid runs with 0 errors. The conditions are P0, the default prompt used throughout the paper; P1, a terser instruction that drops the explicit formatting scafold; P2, an explicit chain-of-thought instruction; and P0+, the default prompt at raised temperature. Every condition is a choice a competent practitioner might defensibly make. None is an adversarial perturbation.

Table 7: Trace alignment for all 25 evaluated systems, sorted by gap. “Sim.” is the mean over gold trace steps of the maximum cosine similarity to any step of the model response. The gap is positive for every variant: responses that reach the correct answer track the human reasoning trace more closely than responses that do not.
<table><tr><td>Model</td><td>n correct</td><td>n incorrect</td><td>Sim. (correct)</td><td>Sim. (incorrect)</td><td>Gap</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>274</td><td>719</td><td>0.8159</td><td>0.7625</td><td>+0.0534</td></tr><tr><td>OpenAI o3</td><td>570</td><td>408</td><td>0.8038</td><td>0.7515</td><td>+0.0523</td></tr><tr><td>VL-Rethinker</td><td>291</td><td>701</td><td>0.8148</td><td>0.7657</td><td>+0.0491</td></tr><tr><td>GPT-5</td><td>633</td><td>361</td><td>0.8095</td><td>0.7644</td><td>+0.0451</td></tr><tr><td>Kimi-VL</td><td>290</td><td>538</td><td>0.8368</td><td>0.7929</td><td>+0.0439</td></tr><tr><td>InternVL3.5-8B</td><td>302</td><td>692</td><td>0.8138</td><td>0.7704</td><td>+0.0434</td></tr><tr><td>Visionary-R1</td><td>261</td><td>680</td><td>0.8023</td><td>0.7595</td><td>+0.0428</td></tr><tr><td>Claude Sonnet 4</td><td>396</td><td>541</td><td>0.8200</td><td>0.7787</td><td>+0.0413</td></tr><tr><td>Qwen2.5-VL-3B-Instruct</td><td>213</td><td>695</td><td>0.7959</td><td>0.7569</td><td>+0.0390</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>400</td><td>555</td><td>0.8225</td><td>0.7840</td><td>+0.0385</td></tr><tr><td>Molmo-7B</td><td>201</td><td>622</td><td>0.7936</td><td>0.7554</td><td>+0.0382</td></tr><tr><td>InternVL3-8B-Instruct</td><td>267</td><td>694</td><td>0.8109</td><td>0.7732</td><td>+0.0377</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>367</td><td>506</td><td>0.8386</td><td>0.8018</td><td>+0.0368</td></tr><tr><td>InternVL3-2B-Instruct</td><td>206</td><td>637</td><td>0.8028</td><td>0.7662</td><td>+0.0366</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>424</td><td>544</td><td>0.8220</td><td>0.7859</td><td>+0.0361</td></tr><tr><td>Gemma-3-12B-it</td><td>346</td><td>650</td><td>0.8107</td><td>0.7747</td><td>+0.0360</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>360</td><td>584</td><td>0.8421</td><td>0.8070</td><td>+0.0351</td></tr><tr><td>Gemini 2.5 Pro</td><td>658</td><td>336</td><td>0.8200</td><td>0.7856</td><td>+0.0344</td></tr><tr><td>Claude Opus 4.6</td><td>622</td><td>375</td><td>0.8279</td><td>0.7936</td><td>+0.0343</td></tr><tr><td>Gemma-3-4B-it</td><td>290</td><td>706</td><td>0.8020</td><td>0.7680</td><td>+0.0340</td></tr><tr><td>Ovis2.5-9B</td><td>361</td><td>625</td><td>0.8201</td><td>0.7886</td><td>+0.0315</td></tr><tr><td>Ovis2.5-2B</td><td>284</td><td>706</td><td>0.8046</td><td>0.7738</td><td>+0.0308</td></tr><tr><td>Skywork-R1V2-38B</td><td>366</td><td>583</td><td>0.8317</td><td>0.8034</td><td>+0.0283</td></tr><tr><td>Skywork-R1V3-38B</td><td>385</td><td>603</td><td>0.8321</td><td>0.8085</td><td>+0.0236</td></tr><tr><td>GLM-4.1V-9B-Thinking</td><td>237</td><td>734</td><td>0.7678</td><td>0.7501</td><td>+0.0177</td></tr></table>

Per-model accuracy is stable, deviating by at most 4.8 points from that model’s own mean across conditions. The top-1 ranking is not: GPT-5 leads under P1 and Gemini 2.5 Pro under the other three, while Claude Opus 4.6 is last in all four. We therefore report that adjacent positions among closely matched systems are not identified by accuracy alone on this benchmark, and treat the grounding, faithfulness, and cost axes as the more informative comparison.

## D Critic Validation Protocol

We sampled 200 model traces balanced across 5 categories, 8 open-weight models, and both outcomes (100 correct, 100 incorrect), and asked whether VisualPRM-8B assigns higher scores to correct reasoning. It separates the two by +0.012 overall (0.608 versus 0.596), and the gap is negative in multi-hop (−0.037) and coreference/occlusion (−0.056), the two categories that demand the most compositional and spatial grounding.

We report this because it bounds how CORE-PRM should be read. CORE is critic-free and carries every headline result in this paper; CORE-PRM is a critic-agnostic template whose validity is inherited from whichever judge is supplied, and with an of-the-shelf 8B critic on unseen domains that validity is currently weak. A stronger multimodal critic would improve CORE-PRM without any change to the benchmark.

## E Category Failure Modes

Expanding Section 6, each category exhibits a distinct and deployment-relevant signature.

Visual coreference / occlusion. Strong systems enumerate candidate entities, apply the spatial constraint to each, and eliminate. Weaker systems show a saliency bias, latching onto the most visually prominent entity regardless of the

referring expression, and a spatial phrase inversion in which relations such as “the man behind the woman in red” have their figure and ground swapped.

Handwriting / OCR. A single misrecognized character propagates through otherwise valid downstream arithmetic, producing a confidently wrong answer with an internally consistent chain. This is the clearest case where final-answer scoring and step-level scoring disagree about what went wrong, and where a correct-looking trace is actively misleading.

Navigation / map reasoning. Landmark identification is generally reliable; path composition is not. Failures concentrate in of-by-one directional steps and left/right confusion at intersections. A second pattern is linguistic shortcutting, where the model infers cardinal orientation from street names rather than from the layout.

Multi-hop. The characteristic pattern is first-hop success with second-hop fabrication: the model resolves the first clue and then introduces an unsupported constraint. Longer chains make this worse, not better, which is consistent with the reflective-token analysis in Section 5.4.

Scientific / medical. Models frequently perceive the visual feature correctly and then hallucinate a plausible but incorrect causal attribution. This category has the highest rate of confidently wrong explanations and is the clearest safety concern for high-stakes deployment, since the stated reasoning reads as authoritative.

## F Qualitative Examples

The figures below show one item per category, with the image, the answer options, the ground-truth rationale, and the full reasoning of four systems side by side with their generated token counts. They are included because the aggregate numbers in this paper describe distributions, and the mechanisms in Section E are easier to assess against concrete items. Every panel is verbatim model output.

![](images/255915facc83c34366e8f435bf5a92d5abac3a36924e60645da79c3623085519.jpg)  
Figure 4: Visual coreference under occlusion. The question asks for the registration of the aircraft partially occluded behind the silver plane in the foreground. Gemini 2.5 Pro resolves it by enumerating the three visible aircraft, reading each registration, and matching against the referring expression. Kimi-VL, Qwen3-VL-30B-Thinking and Ovis2.5-9B all answer N177XV, the registration of the foreground plane. This is the saliency bias described in Section E: the referring expression is never applied, and the most prominent entity is returned instead. Token budgets of 614, 296 and 753 show this is not a matter of insuficient reasoning length.

![](images/45eb148a8d169f1b5ae02255a422f79e91a7008ac003146875b625775e685bf6.jpg)  
Figure 5: Handwriting / OCR reasoning. All five candidate words appear correctly in the handwritten passage, so the answer is “none of the above”, which requires transcribing every option correctly before concluding that nothing is missing. Qwen3-VL-8B-Instruct does so. Gemini 2.5 Pro misreads one word and then constructs an elaborate, internally consistent argument for “based”; Gemma-3-12B and VL-Rethinker each settle on “favour”. This is the cascade pattern: the reasoning is valid given the misperception, so the trace looks sound and the error is invisible without the ground truth.

Multimodal Reasoning Benchmark: Multi-Model Comparison  
![](images/e7225172a08179c6ea69c8b6bbbddadd6f5b078989ac3f2bd1d5d185c4d76673.jpg)  
Figure 6: Navigation / map reading. Gemma-3-12B-It reads the label adjacent to the church and answers correctly in 74 tokens. Kimi-VL (140 tokens), Qwen3-VL-8B-Instruct (172) and Qwen3-VL-30B-Instruct (316) all produce longer, more structured deliberations and all answer “Henry Street”. This item is a compact illustration of why we score cost alongside correctness: the cheapest response here is the only correct one, and a leaderboard ordered by accuracy alone would still rank three of these systems above Gemma overall.

![](images/2721b9cf8e1ccb9b54e3ee0e1bc610e7e3c74d342325e8b4229565941f48125b.jpg)  
Figure 7: Multi-hop reasoning. Answering requires chaining what the scene shows (people playing a Wii boxing game) to a fact about the game (each player needs a remote and a nunchuck) to a count. Ovis2.5-9B, Gemma-3-12B and Skywork-R1V3-38B chain correctly. OpenAI o3 counts three players from the image and propagates that into three controllers, an error at the first hop that the subsequent reasoning faithfully preserves.

Multimodal Reasoning Benchmark: Multi-Model Comparison  
![](images/5f6b3205ed4d446c707c406c4f1f18cf8267f8d498d71011c87be7e12f2a8851.jpg)

Figure 8: Scientific / medical reasoning. The nail finding is Beau’s lines, transverse ridges left by repeated arrest of nail-matrix growth, which point to cyclical chemotherapy. Only OpenAI o3 reaches this. Gemini identifies the pattern as Muehrcke’s lines and reasons fluently to nephrotic syndrome; Gemma-3-12B reasons to psoriasis; Qwen2.5-VL-7B concludes none of the options apply. Each wrong answer arrives with a confident, clinically phrased causal chain. This is the category with the highest rate of confidentl wrong explanations, and the reason we treat it as the clearest safety concern for deployment.  
![](images/291bebca2476c7a25092a7477dca0b9a0cb19f52972e9c4090e5c7d97649504c.jpg)  
Figure 9: Navigation / spatial orientation. The question asks which street is the odd one out and states explicitly that the criterion is not the name. The answer is Wilfred Ave, the only street running north–south while the others run east–west. Only OpenAI o3 checks orientation on the map. Qwen3-VL-8B-Instruct (1,861 tokens), Ovis2.5-9B and Gemma-3-12B all reason from the linguisti distinction between “Court” and “Avenue” despite the hint ruling it out, and Ovis2.5-9B visibly loops before committing. This is the linguistic shortcutting failure, and it is the pattern behind the ∼47% navigation scores in Table 2: the models read the map as text rather than as space.

![](images/b6769650b5f5b233a1ace84840d2a7959115345f0ec02dc7b7ce64c0618a93ae.jpg)  
1,000 items · 20–30 min of annotation per item · 807 items entirely newly authored; 193 medical items reuse an expert-validated answer

Figure 10: The four-stage annotation loop. Duties are separated so that no annotator validates an item whose reasoning trace they wrote, and an item that fails Stage 4 returns to its author for revision instead of being silently accepted or discarded. Stage 2 produces the human reference length $L _ { i } ^ { \mathrm { g t } }$ that CORE uses and the reference content that trace alignment uses.

## G Benchmark Construction

PRISM-BENCH contains 1,000 items, 200 per category. Images are drawn from RefCOCO [9] for coreference and multi-hop, schematic layouts derived from OpenStreetMap for navigation, handwriting scans and IAM for OCR, and MedXpertQA [27] for the medical category.

Figure 10 shows the annotation loop. Three annotators work across four stages, and the assignment of people to stages enforces a separation of duties: the annotator who validates an item is never the annotator who authored its trace. An item that fails validation does not simply get dropped. It returns to its author with the validator’s objection, is revised, and is validated again; only items that pass an independent check are released, and items that cannot be repaired are removed. Each item takes 20–30 minutes of annotation efort across the four stages.

Two points about Stage 2 are worth stating plainly, because they afect how the reference traces should be read. First, where a language model was used to suggest an initial draft of a trace, that draft was mandatorily audited and edited by an expert annotator before acceptance; no trace enters the benchmark as raw model output. Second, the trace is an idealized minimal chain, meaning the shortest logically complete path to the answer, which is what makes its length usable as an eficiency floor in Equation (1).

Question, option, answer, and trace text is newly authored for 807 of 1,000 items (80.7%); only source images are reused. The 193 medical items reuse an expert-validated image, question, and answer, with the trace authored against that verified answer. Internal 5-gram Jaccard overlap across questions is low (mean and median maximum pairwise similarity 0.21 and 0.11), so items are not near-duplicates of one another.

Contamination. Because the source images are public, provenance documentation alone cannot establish that answers did not leak. We therefore ran a behavioral check, reported in Section 5.5: if exposure helped, the most-reused sources would be the easiest. They are the hardest. OpenStreetMap schematics sit at 46.8/47.8 for GPT-5 and Opus 4.6 and COCO/RefCOCO images at 60.6/61.3, while handwriting scans, the source least likely to have been memorized, are easiest at 82.1 and 83.6. Dificulty tracks the reasoning demanded rather than the familiarity of the pixels, which is what one expects when questions are written to ask about occluded content and multi-step spatial composition that no public caption contains.