# VeriScene: Reconstructing Crime Scenes from Legal Evidence via World-Model Agent

Kevin Chuanpu Fu<sup>1</sup>, Yongsen Zheng<sup>1</sup>, Zee Kin Yeong<sup>2</sup>, and Kwok-Yan Lam<sup>1⋆</sup>

<sup>1</sup> Nanyang Technological University, Singapore <sup>2</sup> Singapore Academy of Law, Singapore

Abstract. World models take multimodal inputs like text, photos, and diagrams to generate dynamic scenes in accordance with the laws of physics, thus opening a compelling application: fusing multimodal legal evidence to re-create a crime scene and re-enact how an ofence could have been committed. However, feeding the raw, unorganized evidence into a world model fails in forensic use: it silently drops evidence, glosses over contradictory testimony, and produces motion that violates the evidentiary record. This paper presents VeriScene, an agent that orchestrates the world model: it reconstructs crime scenes from forensic photographs and witness statements of varying reliability, keeping every claim traceable to evidence and every motion physically plausible. VeriScene iteratively fuses the evidence into a cited narrative under an auditing loop, verifies the hypothesized dynamics via probe rollouts in the world model with corrective constraint injection, and renders the ofence as a re-enactment video from a fused keyframe. On a benchmark of 25 crime scenarios across 7 physically-driven case types (139 forensic-style photographs and 65 statements with planted unreliability), VeriScene attains 0.9014 evidence coverage and 0.7217 factual consistency (0–1 scale) on the 20 test scenes, outperforming an end-to-end multimodal-LLM baseline by 20.35% in factual consistency and 34.88% in temporal coherence, while generalizing across four LLM orchestration backends at \$1.82 per scene.

Keywords: World models · Crime-scene reconstruction · LLM agents · AI for justice.

## 1 Introduction

Adjudication rests on evidence of many kinds: scene photographs, seized-object records, identification photos, and witness statements of varying reliability. To weigh them, judges and jurors must mentally re-create how the alleged ofence unfolded—yet such reconstruction is subjective conjecture: statements conflict, evidence scatters across modalities, and a mental picture obeys no objective physical law, while misreading a single physical cue (e.g., a glass-fracture direction) can invert an entire case. World models [6, 14] take multimodal inputs and generate dynamic scenes governed by physics, making them a natural instrument for replacing conjecture with a watchable, physically grounded re-enactment.

In this paper, we propose an agent that orchestrates the world model for crime-scene reconstruction: it automatically fuses the heterogeneous evidence into a coherent, evidence-cited account, and iteratively reasons over the draft in an audit loop that eliminates ambiguity and resolves contradictions before any frame is rendered.

Directly prompting generative models with case material fails here: evidence enters without order or organization, so relations among exhibits—which statement corroborates which photograph, which trace arbitrates which conflict—are never modeled; the output drops evidence, ignores contradictions [10], and favors visual appeal over physical correctness [2], leaving no artifact by which errors can be audited back to their inducing evidence.

Inspired by how investigators draft a hypothesis, cross-examine it against the evidence, and test its physical feasibility, we posit that reconstruction quality is governed by iterative verification rather than model scale: citation audits catch uncovered evidence, world-model probes catch physically invalid dynamics, and verified-physics injection catches unfaithful rendering.

We therefore introduce VeriScene, which turns multimodal legal evidence into physically faithful crime-scene re-enactments: it fuses the evidence into a grounded narrative with per-claim citations, distills a physics-annotated motion specification, and renders the re-enactment video from a fused keyframe—each stage guarded by its own verification loop.

Realizing this pipeline raises one challenge per module: comprehensiveness, met by an auditing agent issuing targeted follow-ups until every evidence item and contradiction is covered; physical faithfulness, met by probing the hypothesized key event in the world model and injecting corrective constraints; and faithful rendering, met by compiling the verified constraints into the generation prompt with completeness and anti-exaggeration clamps.

To assess VeriScene, we construct a benchmark of 25 crime scenarios across 7 physically-driven case types (139 forensic-style photographs, 65 witness statements with planted unreliability). VeriScene achieves 0.9014 evidence coverage and 0.7217 factual consistency (0–1 scale) on 20 test scenarios, outperforming an end-to-end multimodal-LLM baseline by 20.35% in factual consistency and 34.88% in temporal coherence, and generalizing across five LLM backends at \$1.82 per scene.

In summary, the contributions of this paper are three-fold:

– New problem. We formulate crime-scene reconstruction as orchestrating a world model to fuse multimodal legal evidence under physical and evidentiary constraints.

– New system. We design VeriScene, an agent pipeline of iterative evidence fusion, world-model physical reasoning, and physics-annotated re-enactment rendering, each with its own verification loop.

– New benchmark. We construct a 25-scenario, 7-case-type multimodal legal evidence benchmark with five decimal-valued metrics<sup>3</sup>, and validate VeriScene against baselines, ablations, and five LLM backends.

## 2 Problem Statement

Task Setting. An investigator (analyst) must reconstruct the course of an alleged ofence from an evidence set E collected at the scene: (i) forensic photographs depicting the post-incident state and (ii) natural-language witness statements from individual viewpoints. However, E is not a consistent record: photographs capture only static end states, at least one of the two to three statements per scene is partially unreliable and may contradict other evidence, and no ground-truth footage exists, so the reconstruction must be justified by the evidence itself.

Given E, the analyst must produce an evidence-grounded narrative—every asserted event traceable to supporting photographs or statement fragments— and a re-enactment video rendering it as physically plausible motion; when the reconstruction rests on an unresolved contradiction, VeriScene reports the conflicting items rather than silently committing to one account.

Design Goals. VeriScene should meet three requirements: (1) Comprehensiveness. Every item in E enters the reconstruction, since omitted evidence invalidates a legal account. (2) Physical faithfulness. The reconstructed dynamics and rendering should obey physical constraints (e.g., momentum transfer, gravity) that generative video models frequently violate [2]. (3) Verifiability. Every claim and rendered event should be attributable to specific evidence items, with contradictions and reliability judgments reported explicitly for human audit.

## 3 System Overview

Handing the entire evidence set to a multimodal model in one prompt fails on three counts: details appearing in only one photograph are silently omitted, contradicting statements are blended into a fluent narrative instead of weighed against the physical evidence, and text-conditioned generation is not physically grounded [2]. Yet each failure is checkable—dropped evidence by auditing against an evidence inventory, contradictions by cross-examination, and physical violations by probing candidate dynamics in a world model and critiquing the rollout [16]—world model here follows the learned-simulator sense of Ha– Schmidhuber [8] and LeCun [9], in its video-generative branch [14]. VeriScene therefore decomposes reconstruction into one generation–verification loop per failure mode, chained as the three modules in Figure 1.

Iterative Evidence Fusion. VeriScene builds an evidence inventory with an identifier per photograph and statement fragment, a VLM fuses it into a draft scene description, and an auditor agent flags uncovered items, uncited claims, and contradictions, driving bounded revisions until every claim carries supporting identifiers.

![](images/6ed081c7fb7fe0c085271674146ea9697ffb31810fe6a54fee84f0f86bf6ffc6.jpg)  
Fig. 1: Architecture of VeriScene.

World-Model Physical Reasoning. This module tests the fused description’s candidate dynamics physically: each candidate is rendered as a short probe rollout in the world foundation model [14], a physics critic inspects it, and detected violations become constraints injected into the next probe, terminating with validated constraints attached to the narrative.

Re-enactment Rendering. Finally, VeriScene compiles the audited narrative and validated constraints into a physics-annotated prompt rendered by the video world model [6]; inheriting identifiers and constraints, every rendered event remains traceable to its supporting evidence.

## 4 Design of VeriScene

## 4.1 Benchmark Construction

Scenario Schema and Generation. As no public dataset couples multimodal legal evidence with a verifiable physical ground truth, we build 25 synthetic scenarios spanning seven case types (Table 1): 20 held-out evaluation scenes (111 photographs, 53 statements, 33 unreliable) plus a five-scene development split. A frontier LLM (Claude [1]) generates each scenario, under the guidance of legal practitioners on evidence types and investigative conventions, against a programmatically validated schema separating the pipeline-visible evidence bundle from a hidden ground truth (factual summary, 4–6-event timeline, and the key physical fact resolving the case); every case is physical-mechanism-driven with 1–2 distractors.

Evidence Image Synthesis. Each scenario specifies 4–7 photographs of four forensic types—victim/perpetrator identification shots, object close-ups beside a ruler scale, and 2–3 wide scene shots with numbered markers—rendered by an image model [5] under a shared template enforcing sober documentation style and a non-graphic safety sufix. Mirroring forensic practice, persons are photographed after rescue, so every image is individually innocuous while the incident remains recoverable from physical traces (e.g., skid marks).

Table 1: Statistics of the 25 constructed crime scenarios.
<table><tr><td>Case type</td><td>Scenes</td><td>Images</td><td>Person/Obj./Scene</td><td>Statements</td><td>Unreliable</td><td>Avg. words</td></tr><tr><td>Physical altercation</td><td>3</td><td>17</td><td>5/6/6</td><td>6</td><td>3</td><td>189.0</td></tr><tr><td>Arson origin</td><td>3</td><td>16</td><td>3/6/7</td><td>9</td><td>6</td><td>176.8</td></tr><tr><td>Burglary path</td><td>4</td><td>23</td><td>4/10/9</td><td>8</td><td>4</td><td>179.5</td></tr><tr><td>Water incident</td><td>3</td><td>16</td><td>4/6/6</td><td>8</td><td>5</td><td>175.1</td></tr><tr><td>Fall from height</td><td>4</td><td>21</td><td>5/8/8</td><td>11</td><td>7</td><td>197.4</td></tr><tr><td>Property damage</td><td>4</td><td>21</td><td>4/8/9</td><td>11</td><td>7</td><td>178.0</td></tr><tr><td>Traffic incident</td><td>4</td><td>25</td><td>4/11/10</td><td>12</td><td>8</td><td>172.8</td></tr><tr><td>Total</td><td>25</td><td>139</td><td>29/55/55</td><td>65</td><td>40</td><td>181.0</td></tr></table>

Planted Testimony Unreliability. Each scenario carries 2–3 first-person statements (150–250 words, police-record register): one reliable, one with a planted time or position discrepancy, and optionally one contradicting another witness (40 of 65 overall); recorded discrepancies and resolutions test whether conflicts are settled by physical evidence rather than averaged testimony.

## 4.2 Multimodal Evidence Fusion

Structured Fusion Drafting. The first module distills the evidence bundle— never the ground truth—into a structured reconstruction: a frontier VLM (Gemini [5]) receives all photographs, captions, and statements under an arbitration rule that photographic and physical evidence overrides testimony (mirroring investigative practice, which treats photographs with greater objectivity than statements), and returns a narrative with per-claim evidence traceability, a timeline of 4–6 steps citing supporting files, surfaced contradictions with physical resolutions, and a keyframe prompt plus a one-sentence motion hint covering the key event through its outcome.

Audit-and-Refine Loop. The draft is audited by the LLM (Claude), a directed critic in the spirit of self-refinement [16] checking that every file is cited or explicitly irrelevant, every contradiction is physically resolved, and the timeline is physically ordered; it returns at most three directed follow-ups, re-injected append-only, for at most three drafting rounds with early stopping.

Documentation-Protocol Grounding. Because victims are photographed unharmed by protocol, an unprimed fusion model repeatedly read their intact appearance as exculpatory; the prompt therefore declares the acquisition protocol explicitly—otherwise its meaning inverts.

Keyframe Synthesis. The keyframe is rendered by the image model conditioned on up to three original scene photographs, inheriting the documented environment; an open text-to-image fallback [14] covers refusals.

![](images/a3725de09f6e9fd22a5151dfeb8999a246675f0a09d3cecb95a2feed38ee0097.jpg)  
Fig. 2: Probe-and-correct physical verification on a real scene.

## 4.3 World-Model Physical Reasoning

Physics Motion Specification. The second module validates intended motion before expensive rendering: the LLM, prompted as a physical-dynamics expert, maps the narrative, timeline, and motion hint to an 8-second image-to-video specification—a video prompt (who moves, where, at what speed, cause before efect), 3–5 physics constraints (e.g., free-fall acceleration), and 3–5 negative terms naming failure modes to suppress (e.g., teleporting, floating).

Probe-and-Correct Verification. Rather than critiquing text, VeriScene critiques motion (Figure 2): an open video world model (Cosmos [14]) renders a low-resolution probe clip (832×480, ∼2 s) from the keyframe, and the VLM critic—scoped to physics-law violations only [2] (e.g., glass shards flying outward against an inward strike)—turns its severity-ranked verdict into at most two added constraints and two negative terms, merged append-only. The probe is best-efort, falling back to the unrefined specification, never blocking the pipeline.

## 4.4 Scene Re-enactment Rendering

Physics-Annotated Prompt Assembly. The rendering prompt concatenates the video prompt, the probe-refined constraints, and three fixed clamps: completeness (the full event through its physical outcome), anti-cinematic (motion follows the stated physics exactly under fixed-camera documentary framing), and non-graphic (restating the safety constraints of Section 4.1).

Primary Rendering with Safety-Aware Retry. A frontier video world model (Veo 3.1 [6]) generates the 8-second clip image-to-video from the keyframe, with per-model quota failover; when incident semantics trip the hosted safety filter, VeriScene retries once with the content reframed as a fictional re-enactment staged by professional actors.

Open-Model Fallback and Frame Extraction. If both attempts fail, rendering falls back to the open world model [14] at probe resolution but full length, reusing the specification and blacklist; eight uniformly spaced frames are then extracted for the judge of Section 5 and the qualitative storyboard.

## 5 Evaluation

We evaluate VeriScene on crime scenarios spanning seven case types, analyzing reconstruction quality against baselines and ablations (Section 5.2), qualitative fidelity (Section 5.3), and the impact of the orchestration LLM (Section 5.4).

Table 2: Reconstruction quality of VeriScene and baselines.
<table><tr><td>Method</td><td>ECR</td><td>FC</td><td>PP</td><td>TC</td><td>VF</td></tr><tr><td>Text-to-video</td><td>0±0.00</td><td>0.5083±0.26</td><td>0.45±0.11</td><td>0.02±0.09</td><td>0.48±0.10</td></tr><tr><td>Testimony-only T2V</td><td>0±0.00</td><td>0.5483±0.29</td><td>0.44±0.10</td><td>0.09±0.25</td><td>0.49±0.12</td></tr><tr><td>Raw-prompt I2V</td><td>0±0.00</td><td>0.5167±0.28</td><td>0.44±0.10</td><td>0.01±0.04</td><td>0.54±0.11</td></tr><tr><td>Caption-only I2V</td><td>0±0.00</td><td>0.2983±0.21</td><td>0.55±0.15</td><td>0±0.00</td><td>0.58±0.15</td></tr><tr><td>LLM-prompt I2V</td><td>0±0.00</td><td>0.6417±0.19</td><td>0.41±0.04</td><td>0.08±0.13</td><td>0.45±0.09</td></tr><tr><td>E2E-LLM†</td><td>0±0.00</td><td>0.6±0.19</td><td>0.44±0.10</td><td>0.215±0.18</td><td>0.6±0.11</td></tr><tr><td>VERISCENE</td><td>0.9014±0.17</td><td>0.7217±0.20</td><td>0.43±0.07</td><td>0.29±0.29</td><td>0.57±0.10</td></tr></table>

<sup>†</sup>The end-to-end baseline emits no per-claim evidence citations, hence zero coverage by construction.

## 5.1 Experiment Setup

Implementation. We prototype VeriScene in Python on the claude-agentsdk: evidence fusion and self-critique use Gemini (gemini-3.1-pro), keyframes are synthesized by gemini-3-pro-image, and video segments are rendered by Veo 3.1 on its fast tier; when the hosted renderer is unavailable, a Cosmos3- Nano low-resolution fallback runs locally on one NVIDIA H200 GPU; the weak orchestration tier (Section 5.4) runs entirely on this local stack (Cosmos text-toimage keyframes and image-to-video renders), all other tiers as above. The judge is Gemini, decoupled from the orchestration brain to avoid self-preference. All reported runs cover the 20 held-out main scenes (the 5 development scenarios are used only for prompt tuning), and the orchestration-tier study (Section 5.4) uses the first 10, except its weak tier.

Metrics. We score each reconstruction with five metrics on a 0–1 scale: Evidence Coverage Rate (ECR), the fraction of evidence files explicitly cited in the fused timeline (computed programmatically); Factual Consistency (FC), the fraction of ground-truth timeline events reflected in the narrative, capped at 0.5 if the key physical fact is missed; Temporal Coherence (TC), the fraction of adjacent ground-truth event pairs shown in correct order across 8 frames extracted from the rendered video; and Physical Plausibility (PP) and Visual Fidelity (VF), scored 1–5 by the LLM-as-judge protocol [20] and normalized to 0–1.

Baselines. We compare against five naive world-model baselines—Text-to-video (concatenated captions and statements prompted directly to the local world model), Raw-prompt I2V (the same text with the first scene photo as the starting frame), LLM-prompt I2V (one cheap text-only LLM call writes the video prompt), and the single-modality Caption-only I2V and Testimony-only T2V— as well as E2E-LLM, which feeds all evidence into a single multimodal prompt and renders with the same video model; w/o Fusion, which replaces the iterative fusion audit with single-shot fusion; and w/o Reasoning, which removes the world-model probe, so no physical-consistency constraints are injected before rendering.

## 5.2 Overall Performance

Table 2 reports means and standard deviations over the 20 main scenes; Figure 4 exposes the per-scene scores behind them. The central result is the gap over the non-agentic baselines: the naive world-model family collapses on temporal coherence (TC 0.00–0.09—their clips rarely depict events in the evidentiary order) and reaches at most 0.6417 FC, with the single-modality variants confirming both modalities matter (dropping testimonies more than halves FC to 0.2983; dropping photographs costs grounding and order alike), and E2E-LLM produces no evidence citations at all (ECR 0.000), whereas VeriScene grounds its timeline in 90.14% of the evidence files on average and improves FC by 20.3% (0.7217 vs. 0.6000) and TC by 34.9% (0.2900 vs. 0.2150) over E2E-LLM—the traceability that matters most in a legal setting.

![](images/cd12711762398f88c4e6a001fd60bca9b9ce5088fcf55590784e6932f9ca7b0f.jpg)

Fig. 3: Per-metric comparison of VeriScene and baselines by case type.  
![](images/fa10f601809c8b7502de30722fdbe9a4d2859100aee909edd8c9c43ea704ae83.jpg)  
Fig. 4: Per-scene scores of VeriScene across five metrics.

Removing either module individually lowers the scores (e.g., ECR drops from 0.9014 to 0.8436 without the fusion audit); PP and VF hover in a narrow band for all agent variants, as these dimensions are bound by the renderer rather than the orchestration [2].

Process Evidence. Both agentic components stay consistently active: the probe flagged physical violations in all 18 probed scenes (1.61 corrective constraints injected on average), and the fusion audit ran the full 3 rounds in 19 of 20 scenes.

Figure 3 breaks FC, TC, and VF down by case type—VeriScene’s envelope dominates every baseline on FC and TC, where the naive world-model family collapses toward the center—and Figure 4 details every scene: fall and arson scenes score highest, as their dynamics (gravity, flame propagation) are well captured by the probe and faithfully rendered, while water-related scenes are hardest—fluid interactions stress both the physical reasoning and the renderer, depressing PP and TC simultaneously.

S05: At dusk on a rural two-lane road, a light-colored sedan tr. .

object  
object  
scene  
![](images/94f331438ca53641b6b883673e6bcbe0a551cc572bc1269fcf4d5d861e553131.jpg)  
S06: In a covered urban parking garage, a mid-size SUV rolled d. .

![](images/f89f4d117eb15015604e9e2882b0bacb3a192e051a9c16f267a9cc8fb6cf5ce2.jpg)  
S13: Two roommates, Dana Whitlock and Marcus Ferrell, argued la. .  
S14: Farmer Errol Banning entered his barn in the early afterno. .  
S09: An unknown offender entered th occupied apartment during .

![](images/5f963cbaf545ccd82386fffc3bd1e11a4ec33da0c66cbd4926207476f2eacaa6.jpg)  
S11: The reported overnight burglary of the boutique was staged. .  
S16: A wood-frame hay barn on a rura cattle property burned du.

![](images/43219238ef8be34d257fc4719fb68eede5c5ae2fa76da6d85f3ba8aef9c98792.jpg)  
S10: During daylight hours an unknown offender broke into an is.  
S17: A midday fire broke out in the enclosed concrete stairwell. .

![](images/b7b4b02a628c5c77f80663a1f7b8a94fa87ab9b213bd50a72d7e9493674bf16c.jpg)  
S18: A small suburban electronic retail shop burned in the ear. .

![](images/63f12db0842bcd191d621c006ac778197d726475e1b93e4caa9abce570fc098e.jpg)

![](images/4e50241b58a35b525aed1bedf64708fed3527f37ae2d5b92efa621a82d20082a.jpg)  
S23: A wooden post-and-rail boundary fence between two rural pr. .

![](images/0d8b04a7aa217ad89643f8abb02c33a4153be050b30d3aa78480a466be2803e0.jpg)

![](images/d9da4be6cdffabb8896547f4949cdc52dd4b549899ea3a0393b0256a7e72644a.jpg)  
S24: During a nighttime windstorm, a large mature tree standing. .

![](images/3d4fd888490f0a9894ab30501638d093dc1cfd29a704a393802381d03769ce56.jpg)

![](images/6173ccc3bd8c14d2ff28536a83b769ed1c203a9faef81510f1ce2ba5de61ec81.jpg)  
S19: Around 22:40 an elderly heavyset man walked out onto a flo.

![](images/c250b73003754a8e512b79ccb4b9036bbac8a7fb01ff2660d24136a13add3c8a.jpg)  
S22: A silver compact sedan was legally parked on Level 2 of an.  
S21: In the late morning a slightly built teenage boy walked do.

![](images/e1d3df593db56447c8f5ade9766b1f5fe168fd9e7d53679bb3b3c7a3214b1f19.jpg)

![](images/8e18b64128b3111cb016440f0f02707f388aad4bb527a53c12d12281820e573f.jpg)

![](images/48b034310a56d8bac1c3ab59d222d7a237c728af297dec4d25c00b629d8c91e6.jpg)  
(a) Re-enactment key frames of 15 representative scenarios.

![](images/6c3a919117df62838c3f627815fbd1072bdb5868f67075284a2359ebd320db34.jpg)

scene  
![](images/8440287b40890114cfd325c0c9333e9c89420a002dcf02ef2edfa014dae56658.jpg)

![](images/885695107b6c20a49002367b837aa5393a337b3b1042240425aeedce6cdb5e6c.jpg)

![](images/ad35b752b99d321d74e60e9fdc828a886c697187eb6b2726c266edea9e6cb04d.jpg)

![](images/acc6881f19645530218907cbb82c527756e9b75a600cbc9433e72b5a46bbd39f.jpg)

![](images/45b652f4b3223b9f305fd1a137e7afcb4d1dcb71a44b3303f0821113c222027c.jpg)

![](images/d6155da775d389384ef7e2a55b87ed3ad6c1509f4d4d6b2bc8dd4646a05f183d.jpg)

perpetrator  
![](images/6f4581972844c37d625693ce55e4f2b61655e1fc7f44ae648ad2f68b8e195b27.jpg)

(b) End-to-end reconstruction of an arson scenario.  
![](images/44bed35bcaa642e6ecbd47232aa8bec8e6ac099a67e8b4efe55a9b2e3adf11fe.jpg)  
Fig. 5: Qualitative results of VeriScene.

E2E-LLM composite 0.38

VeriScene composite 0.76

![](images/479b31567a0a83e57649c16a9275d567f1a1826a55137c8319981ece9a3aded7.jpg)

![](images/c18c0529980942025b394a93c774dfbe154ba68712fcf4b1ee203cd638fd3982.jpg)

![](images/393de8d24d2cbfbfd119355874e86093332f8ba12141115aa4b4ff3a79cb7399.jpg)

![](images/396e9cb4c8a416251d2553d9284bbe0787309d99524a71840b69210c93edd756.jpg)  
(S16) Ground truth: The V-pattern apex sits at floor level along a linear pour trail with accelerant-positive char, proving an external ignited liquid rather than an internal hay-stack self-...  
E2E-LLM composite 0.34  
VeriScene composite 0.66

![](images/84d91f20e027506642d8b3becaaea02d7436b50ae784878bcc5a398ae2e73f31.jpg)

![](images/62c9de0858cd680c48ec91423e6b26f571d57e041f778573847f6f10ef9bf1ef.jpg)

![](images/abc670f8c49c54cb89bce2c9aeb595bd57d3f786333b7846ebc4a30ea250299c.jpg)  
(S19) Ground truth: The bracket fracture surface is dull, oxidized, and layered with old corrosion product across most of its area with only a thin rim of bright fresh metal, proving a pre-e...

![](images/c395797caa8757cb1db2315a4c5c0d6f16b163c861fd0d9d4c289d97c7df7183.jpg)

![](images/e82bfafb106f2a1196abeb9f2fd60e807140e01f2601e6c657353291b2022fb2.jpg)  
Fig. 6: Re-enactments of E2E-LLM and VeriScene on two scenarios.

## 5.3 Qualitative Analysis

Figure 6 contrasts both pipelines on two scenarios: in the arson case E2E-LLM stages a generic exterior blaze untethered to the exhibits, whereas VeriScene re-creates the evidenced indoor accelerant-trail fire; in the dock case E2E-LLM renders a static, unpopulated harbor while VeriScene stages the documented railing failure and fall—in both, the agent keeps the rendered composite anchored to the ground-truth physical fact.

Table 3: Quality, cost, and latency across orchestration LLM tiers.
<table><tr><td>Brain</td><td>ECR</td><td>FC</td><td>PP</td><td>TC</td><td>VF</td><td>$/scene</td><td>min</td></tr><tr><td>Claude Opus</td><td>0.9</td><td>0.6633</td><td>0.44</td><td>0.375</td><td>0.6</td><td>1.82</td><td>11.0</td></tr><tr><td>Claude Sonnet</td><td>0.8121</td><td>0.6467</td><td>0.42</td><td>0.245</td><td>0.54</td><td>1.65</td><td>8.1</td></tr><tr><td>Gemini Pro</td><td>0.8889</td><td>0.65</td><td>0.46</td><td>0.305</td><td>0.6</td><td>1.28</td><td>4.8</td></tr><tr><td>Gemini Flash</td><td>0.9764</td><td>0.6667</td><td>0.42</td><td>0.22</td><td>0.54</td><td>1.39</td><td>4.9</td></tr><tr><td>Claude Haiku (weak)</td><td>0.9361</td><td>0.6983</td><td>0.4</td><td>0.1175</td><td>0.48</td><td>0.13</td><td>5.8</td></tr></table>

Figure 5a presents one key frame for 15 representative scenarios, showing coherent scene composition across case types from indoor falls to vehicle collisions. Figure 5b walks through the S16 arson case end to end: from a bagged accelerant container and a V-shaped scorch photograph, VeriScene renders the causal chain—pour, ignition, flame racing along the trail, fire climbing into the evidentiary V-pattern—turning disconnected exhibits into a physically ordered narrative a fact-finder can inspect frame by frame.

## 5.4 Impact of the Orchestration LLM

Finally, we swap the orchestration brain across five tiers (Table 3, Figure 7a). Across the four commercial tiers, overall quality is remarkably stable (ECR 0.812–0.976, FC 0.647–0.667, PP/VF within 0.04/0.06)—suggesting the programmatic checks do the heavy lifting and insulate quality from the choice of brain. The dimension most sensitive to brain capability is temporal coherence: Opus attains TC 0.375 versus 0.220 for Flash, so stronger reasoning primarily improves event ordering. The weak tier pushes this to the extreme, pairing the cheapest brain (Claude Haiku) with a fully local renderer at \$0.13 per scene— 14× cheaper than the Opus tier’s \$1.82: the audit mechanism still carries it to 0.936 evidence coverage, but temporal coherence collapses to 0.118 and visual fidelity to 0.480—the checks preserve evidence grounding under weak brains, while temporal quality requires a stronger orchestrator. Part of the weak tier’s VF/PP drop is renderer-bound (it renders via the local world model). Figure 7b decomposes cost and latency by module: with Opus, VeriScene averages \$1.82 per scene at 9.1 minutes, dominated by rendering, while the flash tier is far cheaper at comparable quality—practical for high-volume triage.

## 6 Discussion and Related Work

Extensions: Sketch Maps, 3D Scans, and Audio Evidence. Investigative practice routinely produces a sketch map of the scene—where witnesses stood, where items were found, a vehicle’s path—and increasingly 3D scene scans; both are natural added modalities for VeriScene, serving as spatial priors that anchor evidence positions during fusion and keyframe synthesis. Multimodal LLMs likewise accept audio [5], spoken statements joining the representation with semantic communication [17] easing field transmission; the contradiction ledger exports resolved conflicts as cross-examination points.

![](images/691bb06145e647c98e45da8a2e4d0b4201aacf50bb17364dd2b8903711c95819.jpg)  
(a) Quality and cost across orchestration LLMs.

![](images/fa080de2cbd2ed26a93b5afac3e5845272ccc52bfc269c95e8ab2015f547533d.jpg)  
(b) Per-module cost and wall time.  
Fig. 7: Orchestration-LLM study on the 10-scene subset.

Limitations and Forged Evidence. The renderer imperfectly obeys text-level physics constraints [2]; LLM-as-judge scoring carries biases [20], mitigated by rubrics; the synthetic benchmark (ethics precludes real case files) leaves transfer unverified; and VeriScene models single-event scenes. Generative models can also fabricate evidence, so deployment needs authenticity screening [15, 19, 13]; VeriScene’s physical-consistency checks flag evidence whose dynamics violate world-model constraints.

Legal Status and Ethical Considerations. A rendered re-enactment fits the category of demonstrative evidence: under provisions such as section 68A of Singapore’s Evidence Act [18], it is admissible where it aids the court’s comprehension of the primary evidence—contextualising the photographs and statements it is built from, never substituting for them; VeriScene’s per-claim citations provide the linkage such a tender requires. Persuasive-but-wrong renderings could bias fact-finders, so courtroom use still requires AI-generation disclosure, provenance labels [11, 13], and presentation alongside the primary evidence.

Video Generation and World Models. Difusion-based video generators [3, 6] are increasingly framed as world simulators [14, 4]: trained on Internet-scale video, they internalize approximate scene dynamics and serve as planning substrates for embodied agents. Their objective, however, rewards visual plausibility rather than fidelity to any particular factual record, and physical-commonsense benchmarks expose systematic violations [2]; VeriScene therefore treats the world model as a renderer to be constrained—injecting evidence-derived physics rather than trusting unconstrained rollouts.

Multimodal LLM Agents. Agent frameworks pair vision-language models [10] with tool use and self-refinement [16]; VeriScene specializes them for evidence fusion.

AI for Legal and Forensic Applications. Legal NLP focuses on textual tasks such as judgment prediction and benchmark reasoning [7, 21]; VeriScene is, to our knowledge, the first to render incident scenes as videos from legal evidence. LLM-as-Judge and Physical Plausibility. Strong LLM evaluators track human preference despite biases [20]; VideoPhy [2] and PhyGenBench [12] show generators often violate physical commonsense, whereas our benchmark demands consistency with a factual record.

## 7 Conclusion

This paper presents VeriScene, an LLM-agent pipeline that fuses forensic photographs and witness statements into a structured scene representation, derives world-model dynamics constraints, and renders the event as a re-enactment video, outperforming an end-to-end multimodal-LLM baseline on evidence coverage, factual consistency, and temporal coherence; we hope it establishes a demonstrative aid to the primary evidence—never a substitute for it.

Acknowledgements. This research is supported by the National Research Foundation, Singapore and Infocomm Media Development Authority under its Trust Tech Funding Initiative. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore and Infocomm Media Development Authority.

## References

1. Anthropic: The claude 3 model family: Opus, sonnet, haiku. https://www.anthropic.com/news/ claude-3-family (2024)

2. Bansal, H., et al.: VideoPhy: Evaluating physical commonsense for video generation. In: ICLR (2025)

3. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent difusion models. In: CVPR (2023)

4. Bruce, J., Dennis, M.D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al.: Genie: Generative interactive environments. In: ICML (2024)

5. Gemini Team, Google: Gemini: A family of highly capable multimodal models. arXiv:2312.11805 (2023)

6. Google DeepMind: Veo: Our state-of-the-art video generation model. https://deepmind.google/models/veo/ (2025)

7. Guha, N., et al.: LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In: NeurIPS (2023)

8. Ha, D., Schmidhuber, J.: Recurrent world models facilitate policy evolution. In: NeurIPS (2018)

9. LeCun, Y.: A path towards autonomous machine intelligence. https://openreview.net/forum?id=BZ5a1r-kVsf (2022), accessed: 2026-08-25

10. Liu, H., et al.: Visual instruction tuning. In: NeurIPS (2023)

11. Liu, Z., et al.: Reversible data hiding in encrypted images using adaptive block compression. IEEE TCSVT (2026)

12. Meng, F., et al.: Towards world simulator: Crafting physical commonsense-based benchmark for video generation. In: ICML (2025)

13. Meng, R., et al.: Proactive image manipulation detection and tracing in fake news. IEEE TDSC (2026)

14. NVIDIA: Cosmos world foundation model platform for physical AI. arXiv:2501.03575 (2025)

15. Sha, Z., et al.: DE-FAKE: Detection and attribution of fake images generated by text-to-image generation models. In: ACM CCS (2023)

16. Shinn, N., et al.: Reflexion: Language agents with verbal reinforcement learning. In: NeurIPS (2023)

17. Si, P., et al.: Fine-tunable semantic communication for image transmission. In: BigCom (2024)

18. Singapore Statutes Online: Evidence act 1893, section 68a. https://sso.agc.gov.sg/Act/EA1893 (2012), accessed: 2026-08-25

19. Wang, J., et al.: PD2Net: A prototype-guided generative copy-move forgery image detection and distinguishment framework. IEEE TDSC (2026)

20. Zheng, L., et al.: Judging LLM-as-a-Judge with MT-Bench and chatbot arena. In: NeurIPS (2023)

21. Zhong, H., Xiao, C., Tu, C., Zhang, T., Liu, Z., Sun, M.: How does NLP benefit legal system: A summary of legal artificial intelligence. In: ACL (2020)