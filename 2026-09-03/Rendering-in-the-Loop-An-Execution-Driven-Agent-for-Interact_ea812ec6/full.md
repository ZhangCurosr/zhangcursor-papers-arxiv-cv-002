# Rendering-in-the-Loop: An Execution-Driven Agent for Interactive Web Development

Yilong Guo<sup>1∗</sup>, Hanqi Chen<sup>1∗</sup>, Zixiao Ye<sup>2</sup>, Guanzhong Wang<sup>1†</sup>, Chen Yu<sup>2</sup>, Zeyu Chen<sup>1</sup>

<sup>1</sup>Baidu Inc., Beijing, China

<sup>2</sup>School of Computer Science and Technology, Huazhong University of Science and Technology, Wuhan, China {guoyilong, chenhanqi, wangguanzhong}@baidu.com

## Abstract

Multimodal large language models have achieved remarkable progress in front-end web development, generating interactive webpages from multimodal references such as screenshots and interaction videos. However, existing work largely emphasizes visual metrics such as aesthetics and layout similarity, while overlooking the more critical validation of interactive functionality. We present RILA, an execution-driven agent that puts browser rendering in the loop, iteratively editing generated code from runtime interaction feedback. RILA introduces an Action Interaction Verification (AIV) module that replays the reference interaction trajectory on the generated webpage to collect grounded execution-aware observations, and an Execution-Aware Rendering Score (ERS) that jointly measures interaction correctness and visual fidelity to guide iterative optimization. We further build an execution-verified data synthesis pipeline that produces diverse, high-quality training data, ofering gains complementary to inference-time optimization. On IWR-Bench, RILA consistently improves both interaction and visual fidelity across foundation models. Notably, with our training pipeline, RILA lifts the compact Qwen3.5-9B backbone from 40.40% to 57.52%, surpassing far larger one-shot generators, including the 1T-parameter Kimi-K2.6 (55.61%) and the proprietary GPT-5.5 (55.74%).

## Introduction

Recent advances in multimodal large language models (MLLMs) have significantly advanced web development from diverse inputs, including textual instructions, screenshots, sketches (Li, Zhang, and Yang 2025), and interaction videos (Xiao et al. 2025; Chen et al. 2026). Powered by strong coding models (Hui et al. 2024; Guo et al. 2024), modern systems can synthesize complete web applications consisting of HTML, CSS, JavaScript, and external assets with impressive visual quality, while coding agents further improve generation through iterative tool use and environment interaction (Yang et al. 2024). These advances have made automatic web development increasingly practical for interface prototyping and application development.

Despite this progress, a usable webpage must satisfy two distinct requirements, visual fidelity (how closely the rendered page matches the reference appearance) and interaction correctness (whether the page behaves correctly under user actions), which are often optimized in isolation, and improving one does not guarantee the other. Existing multimodal web development methods optimize static rendering quality through one-shot generation, and dedicated front-end repair approaches (Yuan et al. 2025) audit that rendering against design guidelines; both produce pages that look plausible but break under user actions. Software engineering agents instead repair code from compiler errors, unit tests, or textual logs; lacking any rendering feedback, they can restore functional behavior yet leave the visual result poor. Either way, the decisive failures surface only during runtime interaction, such as broken event bindings, incorrect state transitions, failed resource loading, and unexpected rendering after user operations; such failures erode interaction correctness yet stay invisible to the static screenshots or textual logs on which these methods rely.

![](images/804b7e4c42595d6a4bb5aa0734a86ba12b89bd850aa297bc96961705d2cfa6b8.jpg)  
Figure 1: Final scores on IWR-Bench. RILA improves every backbone over its one-shot output. Notably, the fine-tuned Qwen3.5-9B<sup>†</sup> refined by RILA (57.52) surpasses every oneshot generator, including the proprietary GPT-5.5 (55.74) and the 1T-parameter Kimi-K2.6 (55.61). Green numbers denote the gain over one-shot generation.

To address this, we propose RILA (Rendering-In-the-Loop Agent), an execution-driven agent that closes the loop between browser execution and code editing through runtime interaction feedback. At its core, an Action Interaction Verification (AIV) module runs each generated webpage in a real browser, replays the reference interaction trajectory, verifies every action, and records the rendering state after each step. From these observations, an Execution-Aware Rendering Score (ERS) jointly measures interaction correctness and visual fidelity, and is used to retain the historically best implementation so that unsuccessful edits do not cause degradation. In parallel, a multimodal critique model turns the runtime observations into structured fix suggestions. Guided by this critique, RILA localizes and repairs faulty code with software engineering (SWE) tools, iterating over execution–critique–edit cycles.

Beyond inference-time refinement, we build an executionverified data synthesis pipeline that jointly samples webpage domains, interaction capabilities, and visual styles under compatibility constraints and validates every sample through browser execution, yielding diverse and reliable training data. On IWR-Bench, RILA consistently improves both interaction and visual fidelity across foundation models, and this data proves complementary. Even with only a compact critique model, RILA substantially improves the fine-tuned Qwen3.5-9B over one-shot generation; and under a stronger critique (Kimi-K2.6), this compact 9B model surpasses both the 1T-parameter Kimi-K2.6 and the proprietary GPT-5.5 that generate webpages in one shot (Figure 1), indicating that execution-driven refinement with a strong critique can ofset a large gap in code-model scale.

Our contributions are summarized as follows:

• We propose RILA, an execution-driven agent that puts browser rendering in the loop, refining webpages through an execution–critique–edit cycle driven by runtime interaction feedback rather than static screenshots or textual logs.

• We turn browser execution into a verifiable optimization signal via an Action Interaction Verification (AIV) module that replays reference trajectories and checks each action, and an Execution-Aware Rendering Score (ERS) that measures interaction correctness and visual fidelity to guide critique generation and bestimplementation selection.

• We propose an execution-verified data synthesis pipeline that samples from a compatibility-constrained taxonomy of domains, interaction capabilities, and visual styles and retains only browser-verified pages. Extensive experiments on IWR-Bench show that RILA consistently improves webpage generation quality across backbone models, with this data providing further complementary gains.

## Related Work

## Web Development

Web development with large models is organized by reference modality. From text alone, WebGen-Bench asks an agent to lay out a runnable multi-file codebase and verifies each functionality by driving the deployed site with a navigation agent (Lu et al. 2025). The visual setting maps an image to code: pix2code first mapped a GUI screenshot to markup (Beltramelli 2018), WebSight synthesized screenshot–HTML pairs at scale for training (Laurençon,

Tronchon, and Sanh 2024), Design2Code established automatic rendering-similarity metrics on real webpages (Si et al. 2025), Web2Code paired instruction-tuning data with webpage-understanding questions (Yun et al. 2024), and Sketch2Code accepts rough sketches, resolving their ambiguity by querying a simulated designer (Li, Zhang, and Yang 2025).

Because monolithic generation omits and misarranges elements on complex layouts, later work adds structure. DCGen divides a screenshot into segments, generates code per segment, and reassembles them into a page (Wan et al. 2025); ScreenCoder splits the task across specialized grounding, planning, and generation agents, and reuses that pipeline as a data engine for fine-tuning (Jiang et al. 2025); WAFFLE instead specializes the model itself, with structure-aware attention over the HTML hierarchy and contrastive training that aligns UI images with code (Liang et al. 2025).

A closer line makes generation iterative. UI2Code<sup>N</sup> embeds generation in a loop of execution and visual inspection, and trains on it with a preference objective that ranks rendered candidates against one another (Yang et al. 2026); VisRefiner ties rendered-versus-target discrepancies to the code edits causing them and self-refines with reinforcement learning (Deng, Yao, and Zhang 2026); DesignRepair audits an interface against a Material Design knowledge base along two streams, its code and its rendered page (Yuan et al. 2025). All three read the render rather than exercise it. Interaction is meanwhile treated as an evaluation target, from interactive prototypes (Xiao et al. 2025) and authored interaction paths (Wu et al. 2026) to full user interaction videos (Chen et al. 2026).

## Software Engineering Agents

LLM-based software engineering agents resolve programming tasks by interacting with a development environment. Much design efort concerns the agent–codebase interface: SWE-agent tunes commands and feedback formats to what a model can reliably use rather than to human ergonomics (Yang et al. 2024), CodeAct makes executable code the action space, so actions compose and revise from interpreter output (Wang et al. 2024), and OpenHands consolidates these into an open platform (Wang et al. 2025). For defect resolution, AutoCodeRover navigates program structure with AST-aware search and fault localization (Zhang et al. 2024), Agentless shows a fixed localize–repair–validate pipeline can rival agentic loops (Xia et al. 2024), and AgentCoder pairs a programmer with test-design and testexecution agents that iterate until the tests pass (Huang et al. 2023)—progress measured by execution, on repository test suites (Jimenez et al. 2024) or in procedurally built verifier environments (Jain et al. 2025).

Web agents instead act inside live environments, perceiving rendered pages to operate real websites (Zhou et al. 2024; Deng et al. 2023; Zheng et al. 2024; Lù, Kasner, and Reddy 2024; Xu et al. 2025), without authoring the code behind them. RILA brings execution inside the generation loop: it replays the reference interactions, verifies each action in a real browser, and drives code search and editing with feedback that static renders and textual logs cannot expose.

![](images/a7990678679385e5ef270ee4b2096ee9efa511dcbc234e0f69a27f3c257e9564.jpg)  
Figure 2: Overview of RILA. RILA closes the loop between browser execution and code editing for interactive webpage generation. Starting from an initial implementation generated by a code model, the proposed Action Interaction Verification (AIV) module executes the webpage in a real browser and collects runtime observations. A critique model analyzes the execution results and visual discrepancies to generate fix suggestions, while the proposed Execution-Aware Rendering Score (ERS) maintains the best implementation across iterations to avoid performance degradation. Guided by runtime observations and structured critique, the code model iteratively refines the webpage implementation, progressively improving rendering quality and interaction correctness.

## Method

## Overview

RILA formulates reliable web development as an executiondriven incremental editing problem rather than one-shot generation or regenerate-from-scratch refinement. This design is motivated by two observations: many interaction failures surface only when the webpage is actually executed and are thus invisible to static code inspection or screenshot comparison, and regenerating the entire page after each observed failure tends to overwrite already-correct code, causing regressions and unstable optimization. RILA instead executes the webpage in a real browser, diagnoses the observed failures, and applies localized edits.

As illustrated in Figure 2, given a multimodal task specification $\boldsymbol { x } = \{ \boldsymbol { r } , \boldsymbol { v } , s \}$ , where r denotes the textual instruction, v is the reference interaction video, and s denotes the accompanying resource images (e.g., logos and media assets), a code generation model $G ( \cdot )$ first produces an initial webpage implementation $C _ { 0 } = \dot { G } ( x )$ , the generated frontend project. From the reference video we further derive the reference demonstration $R ,$ comprising the reference screenshots and the interaction trajectory $A = \{ a _ { 1 } , \ldots , a _ { N } \}$ to be reproduced; R serves as the target for both execution and evaluation.

RILA then iteratively optimizes the generated implementation through four tightly coupled components, detailed in the following subsections: Action Interaction Verification (AIV) executes the webpage with Playwright and replays the reference trajectory A to collect runtime screenshots and action execution logs; Execution-Aware Rendering Score (ERS) scores interaction correctness and visual fidelity from these observations and retains the historically best implementation; Execution-Aware Critique turns the observed discrepancies against R into structured fix suggestions; and Tool-Assisted Code Editing uses software engineering tools to localize and incrementally repair faulty code. The same execution-based verification further acts as a quality filter in our Execution-Verified Data Synthesis pipeline.

## Action Interaction Verification

To evaluate the generated webpage under realistic conditions, the Action Interaction Verification (AIV) module executes the generated project in a real browser using Playwright, instead of relying on static screenshot comparison. Given a candidate implementation, AIV first loads the page and analyzes its rendered structure and interactive elements to ground each reference action to a concrete target. It then sequentially replays the reference interaction trajectory $A =$ $\{ a _ { 1 } , \dotsc , a _ { N } \}$ , where each action $a _ { i }$ corresponds to a browser operation such as clicking, typing, or scrolling, interpreting the surrounding page semantics to perform the interaction as a real user would.

During execution, Playwright records the status of each action and captures the rendered webpage after every successful interaction. Once an action fails, subsequent actions are skipped, faithfully simulating real usage where later steps become unreachable once an earlier one breaks. The resulting observation $O _ { t }$ aggregates the captured screenshots and action execution logs. Note that A serves only as the replay script for $\operatorname { A I V } ;$ it is never exposed to the code generation model, which sees only the instruction and the reference screenshots and must therefore infer the intended interactions from appearance alone.

## Execution-Aware Rendering Score

To quantitatively evaluate each implementation, RILA introduces an Execution-Aware Rendering Score (ERS) that jointly measures interaction correctness and visual fidelity under real browser execution. Computed directly from the AIV observations—the per-action verification results and the rendered screenshots captured during trajectory replay— ERS reflects how the webpage actually behaves at runtime rather than its static appearance, and serves as the criterion for selecting which implementation to edit next.

The overall rendering score is defined as

$$
\mathrm { E R S } ( C _ { t } ) = \frac { 1 } { | \mathcal { T } | } \sum _ { S \in \mathcal { T } } S , \quad \mathcal { T } = \{ S _ { \mathrm { i n t } } , S _ { \mathrm { v i s } } \} ,\tag{1}
$$

where $S _ { \mathrm { i n t } }$ measures interaction correctness and $S _ { \mathrm { v i s } }$ measures visual similarity to the reference.

The interaction term is the execution success rate of the predefined action sequence, $S _ { \mathrm { i n t } } = N _ { \mathrm { s u c c e s s } } / N$ , where $N$ is the total number of expected interaction steps and $N _ { s }$ success denotes the number of actions verified as successful by AIV.

The visual term $S _ { \mathrm { v i s } }$ assesses rendering quality from multiple perspectives by aggregating a set of complementary visual metrics M that capture pixel-level similarity, textual consistency, and semantic structure similarity,

$$
S _ { \mathrm { v i s } } = \frac { 1 } { | \mathcal { M } | } \sum _ { S \in \mathcal { M } } S , \quad \mathcal { M } = \{ S _ { \mathrm { S S I M } } , S _ { \mathrm { O C R } } , S _ { \mathrm { S E M } } \} ,\tag{2}
$$

where $S _ { \mathrm { S S I M } }$ is the Structural Similarity Index (SSIM) between rendered and reference screenshots, $S _ { \mathrm { O C R } }$ measures OCR text similarity, and $S _ { \mathrm { S E M } }$ denotes the cosine similarity between image embeddings extracted by a lightweight vision encoder.

Rather than always editing the most recent candidate, RILA edits the historically best implementation. After each round, it computes the ERS of the current candidate and updates the best implementation by ${ \cal C } ^ { * } = $ arg $\operatorname* { m a x } _ { C _ { i } , i \leq t } \mathrm { E R S } ( C _ { i } )$ . The subsequent editing round is then initialized from $C ^ { * }$ rather than the latest webpage, so that unsuccessful edits do not degrade performance. Note that ERS only decides which implementation to carry forward; the editing itself is driven by the structured critique feedback described next.

## Execution-Aware Critique

Given the execution observation $O ^ { * }$ of the current best implementation $C ^ { * }$ and the reference demonstration $R ,$ a multimodal critique model analyzes the discrepancies between the generated webpage and the expected behavior to produce critique feedback $\bar { \Delta _ { t } } = \mathcal { C } ( O ^ { * } , \bar { R } )$ . The comparison spans both interaction and rendering aspects: it examines functional usability, such as whether each action triggers the expected behavior, as well as visual and design quality, including layout structure, information hierarchy, alignment and spacing, style consistency, responsive design, and accessibility.

Rather than emitting a full rewrite, the critique model summarizes the execution results into structured fix instructions, including the identified issue, supporting execution evidence, and actionable suggestions. Such execution-aware feedback transforms low-level browser observations into high-level editing objectives.

## Tool-Assisted Code Editing

Given the structured critique $\Delta _ { t }$ , RILA performs incremental code editing with a tool-assisted SWE module $s .$ , producing the round-t candidate implementation $C _ { t } = \bar { S } ( C ^ { * } , \Delta _ { t } )$ .

This module incrementally edits the webpage by invoking software engineering tools such as semantic code search, file editing, shell execution, and runtime verification within an execution-based environment. Through multiple rounds of tool interaction, it progressively localizes faulty implementations, generates code patches, and validates the edited webpage.

The updated implementation is subsequently re-executed by the AIV module, forming a closed-loop optimization process that runs for $\dot { K }$ rounds.

## Execution-Verified Data Synthesis

Beyond inference-time refinement, we further construct a large-scale training corpus of interactive webpages. Given a sampled task query, a powerful LLM synthesizes a complete webpage, which we then execute and retain only if all its intended interactions run successfully; this execution-verified pipeline yields diverse, functionally valid training data at scale. As illustrated in Figure 3, it runs in four stages. We first sample seeds from a global taxonomy distilled from webinteraction datasets (Deng et al. 2023; Zheng et al. 2024; Lù, Kasner, and Reddy 2024; Xu et al. 2025), screenshots, and design systems, spanning 32 content domains, 110 interaction capabilities, and 308 visual styles, with an LLM-built compatibility matrix filtering implausible combinations. A sampler then expands each compatible combination into a structured generation recipe, leaving the concrete interaction logic for the LLM to improvise at generation time. Typed image placeholders are next filled with assets from open-source libraries, cached locally and rewritten to relative paths so every page is self-contained and renders deterministically ofline. Finally, each page is verified by execution: we replay its full interaction sequence and keep only pages on which every action succeeds, which rejects roughly one-third of the seeds and acts as a quality gate. The retained data is diverse across all three dimensions, with long-tailed visual styles (the top-20 cover only 18%) that we find important for generalization.

![](images/0ab8b67c1611448a64eb83c1fee8694cd1d948641abd76b3c8618ed000e29bac.jpg)  
Figure 3: Our execution-verified data synthesis pipeline. Seeds sampled from a global taxonomy of domains, interaction capabilities, and visual styles are expanded into generation recipes, realized as complete webpages with real assets resolved and injected, and finally filtered by browser execution, which retains only pages whose every intended interaction succeeds.

## Experiments

## Experimental Setup

Benchmark. We evaluate RILA on IWR-Bench (Chen et al. 2026), a benchmark for interactive webpage reconstruction from multimodal inputs. It contains 113 tasks from 100 realworld websites; each provides a textual instruction, a reference interaction video, and the corresponding assets, and requires generating a self-contained HTML webpage that faithfully reproduces both the visual appearance and interactive behaviors of the reference.

Models. We evaluate both proprietary and open-source foundation models, including Kimi-K2.6, GPT-5.5, and Qwen3.5-9B. In addition, we evaluate our supervised finetuned model, denoted as Qwen3.5-9B<sup>†</sup>, obtained by finetuning Qwen3.5-9B on our execution-verified synthetic data, to investigate the complementary efects of training-time adaptation and inference-time execution-driven refinement. During evaluation, diferent models can be assigned as the Critique Model and the Code Model. Without RILA, the code model directly generates webpages in a one-shot manner. With RILA, the critique model analyzes runtime execution results and provides structured fix suggestions to guide iterative code refinement. Unless otherwise specified, RILA uses Kimi-K2.6 as the critique model.

Evaluation Metrics. Following IWR-Bench, all generated webpages are evaluated using its agent-as-a-judge protocol in a real browser environment. We report Interactive Functionality Score (IFS), which measures the success rate of executing the reference action sequence, and Visual Fidelity Score (VFS), which evaluates rendering quality from both low-level visual consistency and high-level semantic alignment. Specifically, VFS consists of the Low-level Fidelity Score (LFS), computed from OCR similarity and DINO feature similarity, and the High-level Fidelity Score (HFS), assessed by a frontier multimodal judge model. The benchmark further reports a Final Score, computed as the weighted combination of IFS and VFS, as the overall evaluation metric. Implementation Details. Unless otherwise specified, RILA performs three rounds of iterative refinement, with webpages executed in a headless Chromium browser via Playwright. The Tool-Assisted Code Editing module is built on the R2E-Gym execution environment (Jain et al. 2025), and each editing round runs for at most 50 tool-use steps. Unless otherwise stated, all experiments use identical prompts and hyperparameters across diferent backbone models.

Training Details. We fine-tune Qwen3.5-9B on the execution-verified synthetic data using the Megatron backend of the ms-swift framework (Zhao et al. 2025) on 16 H800 GPUs. We perform full-parameter fine-tuning of the language backbone while keeping the vision encoder and the vision–language aligner frozen. To accommodate complete front-end projects, we set the maximum sequence length to 65,536 tokens and cap the input image resolution at 921,600 pixels, and compute the loss only on the final response turn.

## Main Results

Table 1 reports the performance of diferent backbone models with and without RILA on IWR-Bench. To our knowledge, no existing refinement method reports results on IWR-

<table><tr><td>Code Model</td><td>IFS ↑</td><td>OCR↑</td><td>DINO↑</td><td>LFS ↑</td><td>HFS ↑</td><td>VFS ↑</td><td>Final ↑</td></tr><tr><td colspan="8">w/o RILA (one-shot generation)</td></tr><tr><td>Kimi-K2.6</td><td>50.85</td><td>56.11</td><td>80.65</td><td>68.38</td><td>65.03</td><td>66.70</td><td>55.61</td></tr><tr><td>GPT-5.5</td><td>50.12</td><td>61.49</td><td>84.05</td><td>72.77</td><td>64.99</td><td>68.88</td><td>55.74</td></tr><tr><td>Qwen3.5-9B</td><td>34.28</td><td>48.63</td><td>73.48</td><td>61.06</td><td>48.34</td><td>54.70</td><td>40.40</td></tr><tr><td>Qwen3.5-9B†</td><td>36.05</td><td>53.79</td><td>72.78</td><td>63.28</td><td>53.69</td><td>58.49</td><td>42.78</td></tr><tr><td colspan="8">w/ RILA (Kimi-K2.6 critique)</td></tr><tr><td>Kimi-K2.6</td><td>63.39</td><td>62.76</td><td>83.28</td><td>73.02</td><td>70.12</td><td>71.57</td><td>65.85 (+10.24)</td></tr><tr><td>GPT-5.5</td><td>58.24</td><td>67.28</td><td>87.44</td><td>77.36</td><td>70.81</td><td>74.08</td><td>63.00 (+7.26)</td></tr><tr><td>Qwen3.5-9B</td><td>46.90</td><td>54.15</td><td>76.52</td><td>65.33</td><td>51.43</td><td>58.38</td><td>50.34 (+9.94)</td></tr><tr><td>Qwen3.5-9B†</td><td>53.91</td><td>60.79</td><td>80.57</td><td>70.68</td><td>61.24</td><td>65.96</td><td>57.52 (+14.74)</td></tr></table>

Table 1: Main results on IWR-Bench. Each model is evaluated under one-shot generation (w/o RILA) and under RILA (w/ RILA), where RILA uses Kimi-K2.6 as the critique model. Parenthesized values report the Final-score improvement over the same model’s one-shot baseline. <sup>†</sup> denotes Qwen3.5-9B fine-tuned on our synthetic data.

Bench, so we compare each backbone against its own oneshot generation and, in the ablation below, against alternative refinement strategies instantiated within RILA. RILA consistently improves webpage generation quality across all evaluated models and evaluation dimensions, demonstrating that execution-driven iterative optimization is efective regardless of the underlying code generation capability. For the 9B backbone, directly applying RILA raises the final score from 40.40% to 50.34% (+9.94) without any additional training, simultaneously improving functional correctness and visual consistency.

Our automatically constructed execution-verified data and RILA are highly complementary. Supervised fine-tuning on our data improves the base model from 40.40% to 42.78%, and applying RILA on top of the fine-tuned model further boosts the final score to 57.52%—a gain of +14.74 over its one-shot baseline and +17.12 over the original 9B backbone. This shows that better initialization from supervised fine-tuning and execution-driven refinement address diferent aspects of webpage generation and combine efectively. Our central finding is that this combination lets a compact 9B model outperform far larger models that do not use RILA: guided by a Kimi-K2.6 critique, the fine-tuned Qwen3.5- 9B<sup>†</sup> surpasses both the 1T-parameter Kimi-K2.6 (55.61%) and the proprietary GPT-5.5 (55.74%) under one-shot generation, indicating that execution-driven refinement with a strong critique can compensate for a large gap in code-model scale.

RILA also benefits the stronger models themselves. Under a Kimi-K2.6 critique, GPT-5.5 improves from 55.74% to 63.00%, while Kimi-K2.6 improves from 55.61% to 65.85%, the best result in Table 1, showing that executionaware browser feedback remains beneficial even for models that already possess strong code generation capabilities.

## Ablation Study

We analyze the key design choices of RILA, including its execution-driven refinement mechanisms (best-so-far selection, execution feedback, and tool-assisted editing), the choice of critique model, and the SWE iteration budget, and we further probe its generalization on Interaction2Code.

<table><tr><td>Best-so- far</td><td>Exec. Feedback</td><td>Tool Editing</td><td>Iter. Refine.</td><td>IFS ↑</td><td>VFS↑</td><td>Final ↑</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>46.90</td><td>58.38</td><td>50.34</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>44.49</td><td>58.05</td><td>48.56</td></tr><tr><td></td><td></td><td>√</td><td>√</td><td>44.03</td><td>54.34</td><td>47.12</td></tr><tr><td></td><td></td><td></td><td>√</td><td>42.47</td><td>52.63</td><td>45.52</td></tr></table>

Table 2: Cumulative component ablation of RILA on the Qwen3.5-9B backbone; a checkmark marks an enabled mechanism (higher is better). Columns abbreviate Best-sofar Selection (carrying the ERS-best candidate into the next round), Execution Feedback (feeding browser-rendered observations to the critique), Tool Editing (localized multi-step edits with SWE tools), and Iterative Refinement (the shared three-round loop). Disabling Tool Editing (last row) instead regenerates the full webpage from the critique each round.
<table><tr><td>Code Model</td><td>Critique Model IFS ↑</td><td>VFS↑ Final ↑</td></tr><tr><td>Qwen3.5-9B</td><td>Qwen3.5-9B† 40.61</td><td>55.48 45.07</td></tr><tr><td>Qwen3.5-9B</td><td>Kimi-K2.6 46.90</td><td>58.38 50.34</td></tr><tr><td>Qwen3.5-9B†</td><td>Qwen3.5-9B† 46.12</td><td>62.45 51.02</td></tr><tr><td>Qwen3.5-9B†</td><td>Kimi-K2.6 53.91</td><td>65.96 57.52</td></tr></table>

Table 3: Efect of the critique model in RILA on IWR-Bench. For each code model, we pair it with a compact critique (Qwen3.5-9B<sup>†</sup>) and a stronger one (Kimi-K2.6). Higher is better.

Each mechanism improves complementary metrics. Table 2 reports a cumulative ablation on the Qwen3.5-9B backbone with the budget fixed at three rounds, and the mechanisms improve complementary axes. Removing bestso-far selection lowers Final by 1.78 (50.34→48.56), almost entirely from a 2.41-point IFS drop (VFS −0.33): its interaction-outcome signal prevents a correct interactive implementation from being overwritten by later edits. Further removing execution feedback (which also disables ERS) instead reduces VFS by 3.71 (58.05→54.34) with little IFS change (−0.46), as the rendered screenshots that expose layout and styling defects are no longer available. Finally, replacing the tool-assisted SWE agent with single-shot critique-conditioned regeneration degrades both axes (IFS −1.56, VFS −1.71; 45.52 Final), since rewriting the whole page discards already-correct regions rather than applying targeted fixes. Together with the diminishing returns of additional SWE iterations (Figure 4), efective refinement thus depends jointly on execution-grounded feedback, the conditioning strategy, and the editing mechanism.

![](images/d35cb907973ce8beef297bb4c7c1e5a08bd8f58be71c8f8b618986e9de418d70.jpg)  
Figure 4: Ablation of the SWE iteration budget on IWR-Bench (113 tasks). We plot the running-best ERS across successive SWE iterations; the marginal gain diminishes rapidly and largely saturates after three iterations.

RILA generalizes across critique models. RILA relies on a critique model to turn execution feedback into actionable fixes, so its capability upper-bounds refinement quality. Table 3 varies the critique model while fixing the code model. A stronger critique (Kimi-K2.6) yields consistently larger gains than a compact critique (Qwen3.5-9B<sup>†</sup>): it raises the Final score from 45.07 to 50.34 for the Qwen3.5-9B code model and from 51.02 to 57.52 for the fine-tuned Qwen3.5-9B<sup>†</sup>. Notably, even a compact critique—using Qwen3.5-9B<sup>†</sup> itself— already improves over one-shot generation (40.40→45.07 and 42.78→51.02), showing that RILA does not require a large critique model, while a stronger critique further widens the gain.

Efect of the SWE iteration budget. We extend the refinement budget to four iterations on IWR-Bench and record the running-best ERS at each step. As shown in Figure 4, it increases monotonically from 0.500 to 0.671 while the marginal gain rapidly diminishes: the fourth iteration adds only about one-tenth of the first, and the first three already recover 95.3% of the total improvement. We therefore adopt three SWE iterations throughout.

Cross-benchmark generalization. To test whether RILA’s refinement transfers beyond IWR-Bench, we evaluate the same agent unchanged on Interaction2Code (Xiao et al. 2025), an independently collected benchmark for reproducing interactive webpages from screenshots, using Kimi-K2.6 to guide the Qwen3.5-9B code model. Unlike IWR-Bench’s continuous multi-step trajectories, Interaction2Code annotates each page with mutually independent single-step transitions (a source state, a trigger, and the resulting destination); we therefore decompose every page into singleinteraction units and cast each into our interaction schema, preserving the original annotation semantics while matching a per-interaction refinement protocol, which yields 375 tasks. We score with the oficial Interaction2Code evaluator, which reports CLIP, SSIM, and text similarity over the full page and the localized interaction region, together with an interaction position-agreement score and an interaction implementation rate. As shown in Table 4, enabling refinement improves every metric, with the largest gains on interaction implementation (+0.027) and text fidelity (full page +0.025, interaction region +0.021), together with a clear gain in interaction localization (position +0.019). RILA thus remains efective under a diferent input modality, a diferent interaction format, and a diferent evaluator, indicating that its execution-driven refinement is not specific to IWR-Bench.

<table><tr><td>View</td><td>Metric</td><td>w/o RILA</td><td>w/ RILA</td><td>∆</td></tr><tr><td rowspan="3">Full page</td><td>CLIP↑</td><td>0.821</td><td>0.839</td><td>+0.018</td></tr><tr><td>SSIM↑</td><td>0.697</td><td>0.707</td><td>+0.010</td></tr><tr><td>Text ↑</td><td>0.766</td><td>0.791</td><td>+0.025</td></tr><tr><td rowspan="5">Interaction</td><td>CLIP↑</td><td>0.728</td><td>0.739</td><td>+0.011</td></tr><tr><td>SSIM↑</td><td>0.550</td><td>0.554</td><td>+0.004</td></tr><tr><td>Text ↑</td><td>0.429</td><td>0.451</td><td>+0.021</td></tr><tr><td>Position ↑</td><td>0.441</td><td>0.460</td><td>+0.019</td></tr><tr><td>IR↑</td><td>0.851</td><td>0.877</td><td>+0.027</td></tr></table>

Table 4: Generalization to Interaction2Code. IR (interaction implementation rate) is computed over all 375 tasks; the remaining similarity metrics are averaged over the 301 tasks whose interaction is triggered by both variants, giving a strictly paired comparison. Higher is better.

## Conclusion

We presented RILA, an execution-driven agent for multimodal web development that runs each generated webpage in a real browser and iteratively improves it through an execution–critique–edit loop. An Action Interaction Verification (AIV) module replays the reference trajectory and verifies every action, an Execution-Aware Rendering Score (ERS) jointly measures interaction correctness and visual fidelity and retains the historically best implementation, and a multimodal critique turns these observations into structured fixes applied with tool-assisted software engineering. We further introduced an execution-verified data construction pipeline that synthesizes diverse, high-quality training data with gains complementary to inference-time refinement. On IWR-Bench, RILA consistently improves both interaction correctness and visual fidelity across backbones; guided by a strong critique, it even lets a compact 9B model surpass far larger open and proprietary models that generate webpages in one shot. By keeping rendering in the loop, RILA turns runtime interaction into a verifiable optimization signal for reliably improving webpage quality, and we hope it ofers a foundation for future execution-driven coding agents.

Beltramelli, T. 2018. pix2code: Generating Code from a Graphical User Interface Screenshot. In Proceedings of the ACM SIGCHI Symposium on Engineering Interactive Computing Systems (EICS), 3:1–3:6.

Chen, Y.; Liu, M.; Shen, Y.; Li, Y.; Huang, T.; Fang, X.; Zheng, T.; Huang, W.; Yang, C.; Fu, D.; Mei, J.; Wu, R.; Zhao, Y.; Wen, L.; Yang, X.; Mao, S.; Lin, Q.; Yu, Z.; Shen, Y.; Qiao, Y.; and Shi, B. 2026. IWR-Bench: Can LVLMs reconstruct interactive webpage from a user interaction video? In International Conference on Learning Representations (ICLR).

Deng, J.; Yao, K.; and Zhang, L. 2026. VisRefiner: Learning from Visual Diferences for Screenshot-to-Code Generation. arXiv:2602.05998.

Deng, X.; Gu, Y.; Zheng, B.; Chen, S.; Stevens, S.; Wang, B.; Sun, H.; and Su, Y. 2023. Mind2Web: Towards a Generalist Agent for the Web. In Advances in Neural Information Processing Systems (NeurIPS).

Guo, D.; Zhu, Q.; Yang, D.; Xie, Z.; Dong, K.; et al. 2024. DeepSeek-Coder: When the Large Language Model Meets Programming – The Rise of Code Intelligence. arXiv:2401.14196.

Huang, D.; Zhang, J. M.; Luck, M.; Bu, Q.; Qing, Y.; and Cui, H. 2023. AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation. arXiv:2312.13010.

Hui, B.; Yang, J.; Cui, Z.; Yang, J.; Liu, D.; et al. 2024. Qwen2.5-Coder Technical Report. arXiv:2409.12186.

Jain, N.; Singh, J.; Shetty, M.; Zheng, L.; Sen, K.; and Stoica, I. 2025. R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents. arXiv:2504.07164.

Jiang, Y.; Zheng, Y.; Wan, Y.; Han, J.; Wang, Q.; Lyu, M. R.; and Yue, X. 2025. ScreenCoder: Advancing Visual-to-Code Generation for Front-End Automation via Modular Multimodal Agents. arXiv:2507.22827.

Jimenez, C. E.; Yang, J.; Wettig, A.; Yao, S.; Pei, K.; Press, O.; and Narasimhan, K. 2024. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? In International Conference on Learning Representations (ICLR).

Laurençon, H.; Tronchon, L.; and Sanh, V. 2024. Unlocking the Conversion of Web Screenshots into HTML Code with the WebSight Dataset. arXiv:2403.09029.

Li, R.; Zhang, Y.; and Yang, D. 2025. Sketch2Code: Evaluating Vision-Language Models for Interactive Web Design Prototyping. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), 3921–3955.

Liang, S.; Jiang, N.; Qian, S.; and Tan, L. 2025. WAFFLE: Fine-tuning Multi-Modal Model for Automated Front-End Development. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 24786–24802.

Lù, X. H.; Kasner, Z.; and Reddy, S. 2024. WebLINX: Real-World Website Navigation with Multi-Turn Dialogue. In International Conference on Machine Learning (ICML).

Lu, Z.; Yang, Y.; Ren, H.; Hou, H.; Xiao, H.; Wang, K.; Shi, W.; Zhou, A.; Zhan, M.; and Li, H. 2025. WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track.

Si, C.; Zhang, Y.; Li, R.; Yang, Z.; Liu, R.; and Yang, D. 2025. Design2Code: Benchmarking Multimodal Code Generation for Automated Front-End Engineering. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), 3956–3974.

Wan, Y.; Wang, C.; Dong, Y.; Wang, W.; Li, S.; Huo, Y.; and Lyu, M. R. 2025. Divide-and-Conquer: Generating UI Code from Screenshots. Proceedings of the ACM on Software Engineering (FSE), 2(FSE): 2099–2122.

Wang, X.; Chen, Y.; Yuan, L.; Zhang, Y.; Li, Y.; Peng, H.; and Ji, H. 2024. Executable Code Actions Elicit Better LLM Agents. In International Conference on Machine Learning (ICML).

Wang, X.; et al. 2025. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. In International Conference on Learning Representations (ICLR).

Wu, F.; Dong, L.; Gao, C.; Chen, Y.; Huang, Y.; Xiao, Y.; and Liao, Q. 2026. Benchmarking Multimodal LLMs on Code Generation for Complex Interactive Webpages. arXiv:2606.00154.

Xia, C. S.; Deng, Y.; Dunn, S.; and Zhang, L. 2024. Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.

Xiao, J.; et al. 2025. Interaction2Code: Benchmarking MLLM-based Interactive Webpage Code Generation from Interactive Prototyping. In Proceedings of the 40th IEEE/ACMInternational Conference onAutomated Software Engineering (ASE), 241–253.

Xu, Y.; Lu, D.; Shen, Z.; Wang, J.; Wang, Z.; Mao, Y.; Xiong, C.; and Yu, T. 2025. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials. In International Conference on Learning Representations (ICLR).

Yang, J.; Jimenez, C. E.; Wettig, A.; Lieret, K.; Yao, S.; Narasimhan, K.; and Press, O. 2024. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. In Advances in Neural Information Processing Systems (NeurIPS).

Yang, Z.; Hong, W.; Xu, M.; Fan, X.; Wang, W.; Cheng, J.; Gu, X.; and Tang, J. 2026. UI2Code<sup>N</sup> : UI-to-Code Generation as Interactive Visual Optimization. In International Conference on Machine Learning (ICML).

Yuan, M.; et al. 2025. DesignRepair: Dual-Stream Design Guideline-Aware Frontend Repair with Large Language Models. In Proceedings ofthe IEEE/ACM 47th International Conference on Software Engineering (ICSE), 2483–2494.

Yun, S.; Lin, H.; Thushara, R.; Bhat, M. Q.; Wang, Y.; et al. 2024. Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track.

Zhang, Y.; Ruan, H.; Fan, Z.; and Roychoudhury, A. 2024. AutoCodeRover: Autonomous Program Improvement. In Proceedings ofthe 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA).

Zhao, Y.; Huang, J.; Hu, J.; Wang, X.; Mao, Y.; Zhang, D.; Jiang, Z.; Wu, Z.; Ai, B.; Wang, A.; et al. 2025. SWIFT: A Scalable Lightweight Infrastructure for Fine-Tuning. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 29733–29735.

Zheng, B.; Gou, B.; Kil, J.; Sun, H.; and Su, Y. 2024. GPT-4V(ision) is a Generalist Web Agent, if Grounded. In International Conference on Machine Learning (ICML).

Zhou, S.; Xu, F. F.; Zhu, H.; Zhou, X.; Lo, R.; et al. 2024. WebArena: A Realistic Web Environment for Building Autonomous Agents. In International Conference on Learning Representations (ICLR).