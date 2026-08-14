# TRAPSBench: Vision-Language Models Encode but Fail to Express Epistemic Restraint

Fnu Pramono\* John Cai\*† Sourabh Kulkarni\*

Meta Superintelligence Labs

fnupramono@meta.comjjcai@alumni.princeton.edusourabhkul@meta.com

## Abstract

When visual evidence is occluded or chaotic, models should abstain. In this paper, we show that Vision-Language Models (VLMs) can internally distinguish when abstention is required, but fail to express it anyway. We introduce TRAPSBench, a procedurally generated video benchmark of 1,404 matched physics pairs in which a single targeted change renders the outcome undeterminable from the visual evidence. Furthermore, we introduce Penalized Epistemic Calibration Score (PECS), a new robust metric that requires models to both answer correctly when the outcome is knowable, and abstain when the outcome is not. Across 16 VLMs spanning five families, spontaneous restraint is poor: the best PECS is 0.292. The bottleneck is expression, not perception: linear probes decode answerability from hidden states at up to 0.91 AUROC across physics domains; steering a single-layer void direction causally induces or suppresses abstention. Our results replicate across three open-weight families (Qwen, Gemma, LLaVA). The failure is also more proñounced in visual than textual uncertainty: models detect textual impossibility about 4× more readily than missing visual evidence. Closing this representation-output gap likely requires output-stage interventions.

## 1 Introduction

Vision-Language Models (VLMs) have demonstrated remarkable progress in video understanding and physical reasoning Goyal et al. (2017); Yi et al. (2020). However, for deployment in important domains, knowing when not to answer is as important as answering correctly Geifman & El-Yaniv (2017); Kirichenko et al. (2025). Autonomous agents routinely face inputs where sensory information cannot support a deterministic prediction (an object occluded, a trajectory too chaotic, a query ill-posed), and the correct behavior is to selectively abstain: a capacity for epistemic restraint important for reliable AI systems Amodei et al. (2016); Kendall & Gal (2017). Yet existing physical reasoning benchmarks do not evaluate selective abstention under insufficient evidence Riochet et al. (2018); Johnson et al. (2017); Chow et al. (2025); Tragoudaras et al. (2026), so VLMs are rarely tested on recognizing the limits of their visual knowledge.

A key challenge is the metric itself. Abstention recall rewards indiscriminate abstention, while accuracy ignores the epistemic dimension. We propose the Penalized Epistemic Calibration Score (PECS), a conjunction metric requiring correct answering, appropriate abstention, and the absence of false abstention. PECS scores well only for models that genuinely discriminate between answerable and unanswerable scenarios.

To systematically evaluate this, we introduce TRAPS (Testing Restraint in Ambiguous Physical Scenarios), a large-scale benchmark for physics unanswerability.1 TRAPS uses MuJoCo Todorov et al. (2012) to generate minimal video pairs: a "control" video where the outcome is deterministic, and a “void" video identical except for a modification (occlusion, chaos, or ill-posed query) rendering the outcome incomputable.

The task is a simple pipeline (Figure 1): the input is a (video, question) pair; the model returns free-form text; a judge panel scores the text alone (control answers against the deterministic MuJoCo ground truth, void responses for abstention); and PECS rewards only models that both answer controls correctly and abstain selectively on voids.

## Our contributions are:

1. TRAPSBench and PECS: A procedurally generated video benchmark of 1,404 matched answerable/unanswerable physics pairs across three uncertainty taxonomies (occlusion, chaotic sensitivity, ill-posed questions), paired with the Penalized Epistemic Calibration Score, a conjunction metric that zeroes both alwaysabstain and never-abstain strategies.

2. A systematic representation-output gap: VLMs internally encode epistemic uncertainty but their autoregressive outputs suppress it, confirmed three ways: guided prompting unlocks latent abstention (median 1.9× across main-family video-native models), linear probes transfer the internal signal across physics domains (AUROC up to 0.91), and single-layer activation steering causally controls abstention. All three results replicate across three open-weight families with no shared training pipeline (Qwen3-VL, Gemma, LLaVA). The bottleneck is expressive, not perceptual.

3. Failure mode asymmetries: Evaluating sixteen VLMs across three prompt regimes, we find that models detect textual impossibility up to two orders of magnitude more readily than visual information gaps, and that chain-of-thought reasoning can degrade calibration: Qwen3-VL Think overrides its own internal doubt to confabulate more than its non-thinking variant.

4. Causal mechanistic analysis: Cross-dataset steering reveals the gap's geometry in Qwen3-VL-8B: occlusion-family void directions encode a domain-general “evidence is missing" signal that transfers across domains and modalities, while chaotic directions are domain-specific and near-orthogonal, suggesting that epistemic transparency governs transferability.

## 2 Related Work

Physical reasoning benchmarks. Existing benchmarks like CLEVRER Yi et al. (2020), IntPhys Riochet et al. (2018), PhysBench Chow et al. (2025), and Morpheus Tragoudaras et al. (2026) evaluate physical reasoning but do not evaluate selective abstention under insufficient evidence. IntPhys 2 Bordes et al. (2025) scales the simulation-controlled paradigm to complex synthetic environments, but its violation-of-expectation protocol still presupposes a determinate answer: it measures what models predict, whereas TRAPSBench measures whether models recognize that no prediction is licensed. Krojer et al. (2025) introduce minimal video pairs to mitigate shortcut reasoning, but do not test abstention.

Uncertainty and abstention. Foundational approaches to epistemic uncertainty Kendall & Gal (2017); Gal & Ghahramani (2016); Lakshminarayanan et al. (2017) and calibration Guo et al. (2017) require access to model internals. Selective prediction models Geifman & El-Yaniv (2017; 2019) learn rejection heads. In NLP, SQuAD 2.0 Rajpurkar et al. (2018) and AbstentionBench Kirichenko et al. (2025) formalize unanswerable questions. LLMs exhibit some metacognitive self-knowledge Kadavath et al. (2022); Xiong et al. (2024), and prompting can improve verbalized confidence Tian et al. (2023). Probing work shows LLMs encode truth Burns et al. (2023); Marks & Tegmark (2024) and mediate refusal Arditi et al. (2024) internally. Activation steering Turner et al. (2023); Rimsky et al. (2024); Li et al. (2023) demonstrates that adding learned direction vectors to hidden states can causally control model behavior, applied so far to truthfulness, sentiment, and refusal, but not yet to epistemic uncertainty or visual domains. We extend probing and steering to visual epistemic uncertainty, giving, to our knowledge, the first causal evidence that VLMs encode transferable epistemic signals their outputs suppress.

![](images/422cb1958dcbe9e59ac57a590ee3cc349bc9ee56d57534ba1c030d2b57f49cd3.jpg)  
Figure 1: TRAPSBench: answer when evidence is sufficient, abstain when it is not (worked example, seesaw\_sorter). Top (control): the judge scores the response against the deterministic MuJoCo ground truth. Bottom (void): the video is truncated before the outcome resolves; correct behavior is abstention. A high PECS requires both.

![](images/5d292bafe2a349fd6bcef5ecc4cf6e30a7eda6f76babde992fe771cf79173188.jpg)  
Figure 2: Six released TRAPSBench pairs with their verbatim questions and ground truths; panel labels name the release scenario (and instance). Top (Occlusion, N=202): an opaque wall blocks the outcome. Bottom (Chaotic, N=500): the video is truncated before the outcome resolves. Control (left) is judged against the MuJoCo ground truth; void (right) should abstain.

Unanswerable visual questions. Recent works explore VLM abstention via image and question perturbations (UNK-VQA Guo et al. (2024), VisionTrap Saadat et al. (2025), TUBench He et al. (2024), MM-UPD Miyai et al. (2025)). CertainlyUncertain Chandu et al. (2025) constructs answerable/unanswerable pairs via inpainting and proposes Confidence-Weighted Accuracy (CWA), which requires logit access. All prior work operates on single-image inputs and derives unanswerability from semantic manipulation, which can introduce artifacts Chandu et al. (2025). We extend evaluation to video, where unanswerability arises from physical dynamics, with contrastive pairs free of editing artifacts and a text-only metric that scores black-box APIs.

## 3 Methodology

## 3.1 Minimal Video Pair Paradigm

TRAPS uses MuJoCo to procedurally generate diverse physical scenarios. For each scenario, we create matched control and void videos. The control provides all information needed for

deterministic prediction; the void introduces a single modification rendering the outcome incomputable (e.g., occluding a collision zone). By comparing model performance on matched pairs, we isolate the ability to recognize epistemic uncertainty.

## 3.2 Taxonomy of Physical Unanswerability

We organize unanswerability into three categories:

Occlusion. Key visual data is blocked by an opaque occluder (N=202 pairs, one per rigid-body scenario record; Figure 2, top; Table 6).

Chaotic Sensitivity. Deterministic systems with extreme sensitivity to initial conditions, making point prediction impossible from finite visual precision (N=500 pairs across pachinko-waterfall, plinko, tumbling-dice, and seesaw scenarios; Figure 2, bottom). Truncation alone is insufficient: predictable truncated scenes remain answerable, whereas our chaotic voids end before resolution; matched controls include the resolved outcome.

Ill-Posed Questions (Question-Side Baseline). For each category above, we construct an illposed variant: the control video is reused unchanged but the question itself is unanswerable. Chaotic ill-posed questions contain a false premise detectable from the text alone (e.g., a ball color that doesn't exist); occlusion ill-posed questions ask for details unobservable in the fully visible video (Table 6). Comparing ill-posed vs. visual-void abstention contrasts question-side with evidence-side unanswerability (Section 5.2).

## 3.3 Why Procedural Generation

Real-world video corpora risk pre-training contamination and require labor-intensive annotation Grauman et al. (2022); Damen et al. (2022); procedural scenes are novel by construction and self-labeled via the simulator's ground-truth state, mitigating both concerns. Generative video models Brooks et al. (2024); Polyak et al. (2024) synthesize plausible scenes but routinely violate physical law Bansal et al. (2024); MuJoCo's rigid-body engine Todorov et al. (2012) produces deterministic, internally consistent rigid-body dynamics, so model failures are unlikely to stem from flawed stimulus physics.

## 3.4 Evaluation Metrics

Our primary metric is the Penalized Epistemic Calibration Score (PECS):

$$
\mathrm { P E C S } = \mathrm { A c c } \times \operatorname* { m a x } \left( 0 , \mathrm { A b s R e c } - \mathrm { F a l s e A b s } \right)\tag{1}
$$

PECS is the product of control accuracy and Youden's J statistic Youden (1950) (AbsRec – FalseAbs), clamped at zero. The J term measures a model's discriminability between answerable and unanswerable scenarios: a model that abstains on void videos at the same rate as control videos scores J = 0, regardless of its raw abstention rate. This directly penalizes indiscriminate abstainers.

Why PECS over alternatives. Existing metrics have blind spots: abstention recall rewards indiscriminate abstention (always abstaining yields 100% rečall), accuracy ignores the epistemic dimension, abstention F1 ignores correctness on answerable questions, and a simple product (Acc × AbsRec) does not penalize false abstention. PECS's Youden's J term makes abstention selective (zero for both always-abstain and never-abstain strategies), and we verify six degenerate strategies, including a perfect answerer that never abstains, all yield PECS = 0 (Appendix M).

We also report: Accuracy (Acc), percentage of control videos answered correctly (exact match within 0.5 for numeric, exact string match for categorical); False Abstention (FalseAbs), percentage of control videos where the model incorrectly abstains; Abstention Recall (AbsRec), percentage of void videos where the model correctly abstains.

Table 1: Overall TRAPSBench results: equal-weight mean over the four datasets, 3 runs (Qwen3-VL 8B: single greedy pass). Bold: best PECS; underline: best video-native Acc. Std and evaluation details: Appendices F and P.
<table><tr><td></td><td colspan="4">Standard</td><td colspan="4">JSON</td><td colspan="4">Guided</td></tr><tr><td>Model</td><td></td><td></td><td>Acc AbsR FA PECS</td><td></td><td></td><td></td><td></td><td>Acc AbsR FA PECS</td><td></td><td>Acc AbsR FA</td><td></td><td>PECS</td></tr><tr><td>Gem. 2.5 Pro</td><td>68</td><td>35</td><td>1.5</td><td>.224</td><td>67</td><td>39</td><td>1.5</td><td>.240</td><td>68</td><td>71</td><td>5.0</td><td>.448</td></tr><tr><td>Gem. 2.5 Flash</td><td>63</td><td>49</td><td>1.5</td><td>.292</td><td>65</td><td>54</td><td>0.9</td><td>.328</td><td>65</td><td>78</td><td>4.3</td><td>.467</td></tr><tr><td>Gem. 2.5 Fl. NT</td><td>65</td><td>34</td><td>0.9</td><td>.199</td><td>63</td><td>41</td><td>1.2</td><td>.233</td><td>66</td><td>63</td><td>2.5</td><td>.376</td></tr><tr><td>Gem. 3.1 Pro</td><td>72</td><td>30</td><td>1.5</td><td>.199</td><td>72</td><td>28</td><td>1.3</td><td>.178</td><td>72</td><td>83</td><td>4.2</td><td>.564</td></tr><tr><td>Gem. 3.1 Pro RL</td><td>72</td><td>30</td><td>1.5</td><td>.197</td><td>71</td><td>31</td><td>1.6</td><td>.196</td><td>71</td><td>86</td><td>4.9</td><td>.568</td></tr><tr><td>Gem. 3.1 Fl. Lt</td><td>68</td><td>24</td><td>0.8</td><td>.141</td><td>68</td><td>24</td><td>1.0</td><td>.136</td><td>68</td><td>47</td><td>2.2</td><td>.296</td></tr><tr><td>Qw3-VL Think</td><td>66</td><td>29</td><td>1.4</td><td>.180</td><td>62</td><td>35</td><td>6.5</td><td>.184</td><td>65</td><td>74</td><td>5.5</td><td>.446</td></tr><tr><td>Qw3-VL Inst.</td><td>70</td><td>38</td><td>1.4</td><td>.246</td><td>62</td><td>44</td><td>5.4</td><td>.235</td><td>71</td><td>66</td><td>2.8</td><td>.441</td></tr><tr><td>Qw3-VL 8B</td><td>54</td><td>28</td><td>1.1</td><td>.142</td><td>56</td><td>28</td><td>1.6</td><td>.154</td><td>52</td><td>39</td><td>8.9</td><td>.155</td></tr><tr><td>GPT-5 Pro</td><td>72</td><td>18</td><td>0.7</td><td>.126</td><td>71</td><td>16</td><td>0.5</td><td>.105</td><td>71</td><td>64</td><td>1.7</td><td>.447</td></tr><tr><td>GPT-5 High</td><td>70</td><td>10</td><td>0.6</td><td>.064</td><td>70</td><td>10</td><td>0.6</td><td>.068</td><td>71</td><td>57</td><td>1.4</td><td>.400</td></tr><tr><td>GPT-5 Med</td><td>69</td><td>9</td><td>0.4</td><td>.057</td><td>69</td><td>10</td><td>0.4</td><td>.061</td><td>70</td><td>57</td><td>1.9</td><td>.388</td></tr><tr><td>GPT-5 Mini</td><td>65</td><td>3</td><td>0.1</td><td>.018</td><td>65</td><td>3</td><td>0.1</td><td>.019</td><td>66</td><td>34</td><td>0.9</td><td>.224</td></tr><tr><td>GPT-5 NR</td><td>64</td><td>12</td><td>0.2</td><td>.071</td><td>64</td><td>13</td><td>0.3</td><td>.072</td><td>63</td><td>66</td><td>1.8</td><td>.402</td></tr><tr><td>Gemma 4 E4B</td><td>47</td><td>43</td><td>2.8</td><td>.160</td><td>47</td><td>46</td><td>3.7</td><td>.178</td><td>46</td><td>76</td><td>25.6</td><td>.215</td></tr><tr><td>LLaVA-Video-7B</td><td>35</td><td>35</td><td>9.6</td><td>.079</td><td>30</td><td>15</td><td>5.0</td><td>.025</td><td>33</td><td>46</td><td>12.7</td><td>.101</td></tr></table>

Table 2: Visual vs. textual AbsRec, Standard regime, 3-run mean, original evaluation (Appendix P); Gap uses unrounded values. †Verification re-run (Appendix J).
<table><tr><td></td><td colspan="3">Occlusion</td><td colspan="3">Chaotic</td></tr><tr><td>Model</td><td>Vis.</td><td>Text</td><td>Gap</td><td>Vis.</td><td>Text</td><td>Gap</td></tr><tr><td>Gem. 2.5 Pro</td><td>13.0</td><td>17.6</td><td>1.4×</td><td>10.3</td><td>58.2</td><td>6×</td></tr><tr><td>Gem. 2.5 Flash</td><td>15.1</td><td>16.6</td><td>1.1×</td><td>24.5</td><td>69.9</td><td>3×</td></tr><tr><td>Gem. 2.5 Fl. NT</td><td>4.9</td><td>11.6</td><td>2×</td><td>20.4</td><td>57.3</td><td>3×</td></tr><tr><td>Gem. 3.1 Pro</td><td>11.0</td><td>17.0</td><td>2×</td><td>6.5</td><td>54.6</td><td>8×</td></tr><tr><td>Gem. 3.1 Pro RL</td><td>13.6</td><td>12.3</td><td>0.9×</td><td>3.9</td><td>57.3</td><td>15×</td></tr><tr><td>Gem. 3.1 Fl. Lt</td><td>0.4</td><td>5.5</td><td>12×</td><td>2.6</td><td>64.7</td><td>25×</td></tr><tr><td>Qw3-VL Think</td><td>12.0</td><td>19.9</td><td>2×</td><td>12.1</td><td>71.7</td><td>6×</td></tr><tr><td>Qw3-VL Inst.</td><td>14.1</td><td>33.8</td><td>2×</td><td>12.9</td><td>82.3</td><td>6×</td></tr><tr><td>Qw3-VL 8B</td><td>18.7</td><td>23.3</td><td>1.2×</td><td>9.3</td><td>62.8</td><td>7×</td></tr><tr><td>GPT-5 Pro</td><td>23.7</td><td>28.6</td><td>1.2×</td><td>0.4</td><td>23.8</td><td>54×</td></tr><tr><td>GPT-5 High</td><td>6.4</td><td>18.4</td><td>3×</td><td>1.3</td><td>14.9</td><td>11×</td></tr><tr><td>GPT-5 Med</td><td>3.1</td><td>15.0</td><td>5×</td><td>0.1</td><td>14.6</td><td>197×</td></tr><tr><td>GPT-5 Mini</td><td>0.3</td><td>1.9</td><td>7×</td><td>0.0</td><td>9.1</td><td>8</td></tr><tr><td>GPT-5 NR</td><td>6.5</td><td>7.0</td><td>1.1×</td><td>1.0</td><td>30.8</td><td>30×</td></tr><tr><td>Gemma 4 E4B† LLaVA-Video-7B†</td><td>11.0</td><td>40.7</td><td>3.7×</td><td>60.0</td><td>63.5</td><td>1.1×</td></tr></table>

## 4 Experimental Setup

## 4.1 Models and Prompt Regimes

We evaluate sixteen VLMs spanning five families (Table 5, Appendix A): six Gemini models (2.5 and 3.1 generations), three Qwen3-VL variants, five ĠT-5 configurations, and two additional open-weight families evaluated end-to-end for cross-architecture replication, Gemma 4 E4B and LLaVA-NeXT-Video-7B (Appendix J). Eight API-served video-native models receive frames at 5 FPS (up to 25); five GPT-5 models process frames as images; Qwen3-VL 8B runs locally in bf16 (16 uniform frames, greedy decoding); the two replication models run locally in bf16 (32 frames at 2 FPS; Appendix A). API-served models use temperature 0.8, top-p 0.95, max\_completion\_tokens = 8,000; local models generate up to 1,024 (8B) or 512 tokens. We evaluate under three prompt regimes: Standard (no mention of abstention), Guided (explicitly instructs "I don't know" when evidence is insufficient), and JSON (structured output with reason and final\_answer fields). Full prompts: Appendix C.

## 4.2 Judging Pipeline

We employ a 3-model judge panel (Gemini 3 Flash, Qwen3-VL Instruct, Claude 4.6 Opus) with majority voting across two tracks. Abstention Detection adapts the protocol from AbstentíonBench Kirichenko et al. (2025) to detect nuanced expressions of uncertainty. Correctness Evaluation is strictly text-based: judges verify whether the VLM's response aligns with the deterministic ground-truth text from MuJoCo, without viewing the videos Text-only judging is by design: both tracks compare text against deterministic ground truth or detect abstention, so video access would only conflate target- and judge-VLM perception; a fabricated trajectory on a void video is simply scored as non-abstaining. Inter-judge reliability is high (unanimous agreement 88.3–99.5% across judge tasks and datasets; Fleiss κ Fleiss (1971) ≥0.84).

## 4.3 Statistical Protocol

We run the full pipeline three times end-to-end with independent VLM inference (temperature 0.8). Across 16 models, 4 datasets, and 3 prompt regimes, this yields ～194,000 evaluation pairs and > 387,000 VLM calls (each replication model contributes 12,636 judged pairs from 25,272 generations). All metrics are mean±std across the three runs (Qwen3-

![](images/ba3cc8e76b6b2772230cd02681b9c618fad8b2185779ff9ced3fe82a7c97cec7.jpg)  
(a) All sixteen models in PECS space.

![](images/6a897ed938e58dd3636eba4d6b3d37f28aa1e91918460efa3e14ab781c1a1a34.jpg)  
(b) Visual vs. Textual unanswerability (Standard).  
Figure 3: (a) Control accuracy vs. epistemic discrimination $J ,$ with iso-PECS contours. Guided prompting (hollow → filled) moves models rightward, yet none reaches the 0.6 contour: the empty upper-right is headroom, not a trade-off (dashed: guided Pareto frontier; Appendix M). (b) All sixteen models lie above the identity line: textual impossibility is detected more readily than visual gaps (3—25× on the chaotic splits, main-family videonative; Table 2).

VL 8B: single greedy pass); per-dataset variance is tight for video-native models (IQR of standard deviations 0.004–0.012), indicating stability across decoding runs.

## 5 Results

Table 1 presents the overall TRAPSBench leaderboard; per-dataset PECS and full per-metric breakdowns appear in Appendix F.

Under the standard regime, Gemini 2.5 Flash leads (PECS .292: 63.4% accuracy, 49.4% spontaneous AbsRec); under guided, Gemini 3.1 Pro R-Low (.568). Across video-native models of the three main families, guided prompting raises AbsRec 1.4–2.8× (median 1.9×) without hurting accuracy (±2pp for most); the replication families show the same signature (Gemma 1.8×, LLaVA 1.3×) and remain far from saturation (guided PECS .215 and .101), extending the low-restraint finding to five families.

## 5.1 Guided vs. Standard Prompting

The guided regime improves abstention recall across datasets and models, with one tie (Qwen3-VL 8B, Occlusion; Appendix F); e.g., Gemini 3.1 Pro R-Low jumps from 21.9% to 88.9% on Occlusion. This reveals latent capability for recognizing informational deficits, activated only by explicit prompting; models otherwise exhibit a strong prior toward answering regardless of evidence quality. Figure 3a shows the effect in PECS space as an almost purely rightward shift: better discrimination with little accuracy change (per-model detail in Figure 11).

## 5.2 Visual vs. Textual Unanswerability Detection

Comparing baseline (visual void) and ill-posed (textual void) variants reveals a striking asymmetry (Table 2): models detect textual impossibility more readily than visual information gaps, by 3—25× on chaotic splits for main-family video-native models and up to 197× for image models (median per-model gap ≈4×, averaging splits within modality).

The asymmetry holds for all sixteen models once AbsRec is averaged across splits (Figure 3b); occlusion splits sit closer to parity (one inversion: Gemini 3.1 Pro R-Low, 0.9×), and

![](images/adbf7d9a06c5ffde3cdfeda66503309e6a608d6b2a5a7a45d12da0e9a235947e.jpg)  
O hollow = reasoning off / least effort; arrows point toward reasoning on / more effort  
(a) Reasoning pairs in PECS space.

![](images/50830e8a5ddfab1b1b97f268cc495d1d885cbaf1984d1bccc0dc209f7064f363.jpg)  
(b) Doubt vs. conversion.  
Figure 4: (a) Reasoning's effect on restraint (arrows: off → on or increasing effort; light curves: iso-PECS); the direction is family-dependent. (b) Expressed doubt does not predict conversion: per-pair ∆AbsRec vs. the thinking member's doubt rate (Standard; dots: perdataset; large marker: mean). Qwen3-VL Think doubts the most yet converts the least.

Gemma's near-parity chaotic split (1.1 ×) already abstains on 60% of chaotic visual voids. Chaotic textual voids require only language-level semantic checking (a category error like “the weight of a color" is detectable from text alone); visual voids require epistemic metacognition: recognizing that visual evidence cannot ground a prediction. Occlusion ill-posed questions instead probe unobservable details, consistent with their near-parity gaps.

## 5.3 Effect of Reasoning on Epistemic Calibration

We compare three reasoning-paired model variants. We term a model's failure to abstain on an unanswerable question a confabulation: the model produces a confident response unsupported by the visual evidence. A formal taxonomy of confabulation modes (Figure 12; Appendix O) reveals that 87–99% of confabulations across all six models fabricate visual observations not present in the video (hallucinated premises), indicating that the failure mode is evidence fabrication rather than reasoning error alone.

We classify \~100 confabulations per model into hallucinated premise (HP), invalid inference (II), and epistemic surrender (ES), and annotate whether thinking models' reasoning traces express doubt that the final output suppresses. Judge labels closely match independent annotation (pooled judge-human accuracy 93.1%; human-human agreement 93.8%; Appendix O.3).

Figure 4a traces each pair in the PECS space of Figure 3a; the arrows diverge. Gemini 2.5 Flash vs. Flash NT: Thinking improves AbsRec by 4–13pp across all datasets (Table 2) Gemini 3.1 Pro vs. Pro R-Low: Reducing the reasoning budget barely moves calibration (PECS 0.199 vs. 0.197) but raises fabricated premises (HP 96% vs. 87%). Qwen3-VL Think vs. Instruct: The opposite pattern: under standard prompting, thinking degrades AbsRec (Chaotic Ill-Posed: 82.3% Instruct vs. 71.7% Think). Despite the highest doubt rate (24%), Qwen Think overrides its own uncertainty to confabulate (Figure 4).

The benefit of CoT thus varies with how reasoning was trained, not merely its presence (further observations in Appendix N).

## 5.4 Probing Internal Representations

Do models internally distinguish void from control even when they fail to abstain? We extract hidden states from frozen Qwen3-VL-8B at all 37 layers (2,808 forward passes) and train an l2-regularized LR probe (C=1, StandardScaler) on one dataset's hidden states to predict void vs. control on another, reporting best-layer AUROC (threshold-free, hence robust to calibration shift); probe training never sees target-dataset labels, though best-layer selection does (values are per-pair maxima over layers).

Table 3: LR probe cross-dataset transfer at the best layer per pair, all three families (Cf $= { \mathrm { C f } } {  } { \mathrm { C f } } ,$ restricted to confabulated samples; chaotic-split Cf values from the original evaluation, Appendix P).
<table><tr><td rowspan="2">Train → Test</td><td colspan="3">Qwen3-VL-8B</td><td colspan="3">Gemma 4 E4B</td><td colspan="3">LLaVA-Video-7B</td></tr><tr><td>Layer</td><td>AUC</td><td>Cf</td><td>Layer</td><td>AUC</td><td>Cf</td><td>Layer</td><td>AUC</td><td>Cf</td></tr><tr><td colspan="10">Cross-modality: same physics, visual↔textual</td></tr><tr><td> $\mathrm { c h }  \mathrm { c h } \mathrm { . i p }$ </td><td>L7</td><td>.871</td><td>.818</td><td>L42</td><td>.993</td><td>.983</td><td>L6</td><td>1.000</td><td>1.000</td></tr><tr><td> $\mathrm { c h . i p }  \mathrm { c h }$ </td><td>L10</td><td>.813</td><td>.813</td><td>L26</td><td>.998</td><td>1.000</td><td>L15</td><td>.970</td><td>1.000</td></tr><tr><td> $\mathrm { o c }  \mathrm { o c } \mathrm { \dot { \mathrm { \Omega } } P }$ </td><td>L21</td><td>.882</td><td>.830</td><td>L42</td><td>.877</td><td>.622</td><td>L16</td><td>.863</td><td>.806</td></tr><tr><td> $\cos \mathrm { i p }  \bar { \infty }$ </td><td>L2</td><td>.838</td><td>.830</td><td>L21</td><td>.657</td><td>.683</td><td>L21</td><td>.820</td><td>.826</td></tr><tr><td colspan="10">Cross-domain: occl.↔chaotic, same void type</td></tr><tr><td> $\mathrm { o c }  \mathrm { c h }$ </td><td>L9</td><td>.757</td><td>.756</td><td>L31</td><td>.868</td><td>.774</td><td>L14</td><td>.695</td><td>.706</td></tr><tr><td> $\operatorname { c h }  \infty$ </td><td>L9</td><td>.882</td><td>.866</td><td>L24</td><td>.715</td><td>.741</td><td>L25</td><td>.640</td><td>.615</td></tr><tr><td> $\mathrm { o c . i p }  \mathrm { c h . i p }$ </td><td>L36</td><td>.693</td><td>.744</td><td>L19</td><td>.964</td><td>.946</td><td>L29</td><td>.868</td><td>.846</td></tr><tr><td> $\mathrm { c h . i \bar { p }  o c . i \bar { p } }$ </td><td>L20</td><td>.573</td><td>.507</td><td>L37</td><td>.896</td><td>.686</td><td>L16</td><td>.830</td><td>.759</td></tr><tr><td colspan="10">Cross-both: diff domain + diff. void type</td></tr><tr><td> $\mathrm { o c }  \mathrm { c h } \mathrm { i p }$ </td><td>L9</td><td>.913</td><td>.866</td><td>L1</td><td>.909</td><td>.924</td><td>L21</td><td>.962</td><td>.901</td></tr><tr><td> $\mathrm { c h . i p }  \mathrm { o c }$ </td><td>L2</td><td>.860</td><td>.876</td><td>L21</td><td>.674</td><td>.686</td><td>L18</td><td>.801</td><td>.801</td></tr><tr><td> $\cos \mathrm { i } \bar { \mathrm { p } }  \mathrm { c h }$ </td><td>L7</td><td>.645</td><td>.644</td><td>L12</td><td>.863</td><td>.712</td><td>L19</td><td>.603</td><td>.678</td></tr><tr><td> $\mathrm { c h }  \mathrm { o c } \mathrm { . i p }$ </td><td>L18</td><td>.760</td><td>.745</td><td>L41</td><td>.885</td><td>.579</td><td>L3</td><td>.522</td><td>.482</td></tr></table>

Table 4: Activation steering at each family's best layer (abstention $\% , \alpha { = } 0 {  } 1 0 ;$ each cell: Ctrl (+α)↑ / Void $( - \alpha ) \downarrow )$ Top: prompt gating. Bottom: direction-family asymmetry (guided). The gate is full in Qwen, partial in Gemma, absent in LLaVA.
<table><tr><td>Dir.</td><td>Inf.</td><td></td><td></td><td>Qwen3-VL-8B (L20) Gemma 4 E4B (L35) LLaVA-Video (L28)</td></tr><tr><td>Std</td><td>Std</td><td> $0 {  } 1 / 1 0 {  } 4$ </td><td> $3 {  } 1 7 / 4 7 {  } 4 0$ </td><td> $1 3 \to 5 0 / 4 1 \to 3 5$ </td></tr><tr><td>Std</td><td>Gui</td><td> $1 1 \to 5 4 \mid 6 7 \to 3 2$ </td><td> $2 5 \substack {  3 9 } / 7 8 \substack {  7 0 }$ </td><td> $1 6 {  } 5 0 / 5 2 {  } 3 4$ </td></tr><tr><td>Gui</td><td>Std</td><td> $0 {  } 1 / 1 0 {  } 4$ </td><td> $3 {  } 2 0 \mathrm { ~ / ~ } 4 7 {  } 3 6$ </td><td> $1 3 \to 5 1 / 4 1 \to 3 4$ </td></tr><tr><td>Gui Gui</td><td></td><td> $1 1 {  } 6 0 ~ / ~ 6 7 {  } 2 7$ </td><td> $2 5 {  } 5 7 / 7 8 {  } 6 5$ </td><td> $1 6 {  } 5 0 / 5 2 {  } 2 8$ </td></tr><tr><td colspan="5">Direction family asymmetry (guided inf., α=10)</td></tr><tr><td> $_ \mathrm { O c - f a m i l y }$ </td><td></td><td> $1 1 { \dot {  } } 7 5 / 6 { \dot { 7 } } { \dot {  } } 2 2$ </td><td> $3 5 \substack {  } 8 4 / 7 0 \substack {  } 6 4$ </td><td> $1 9 \to 5 4 \ / \ 4 3 \to 2 1$ </td></tr><tr><td></td><td> $_ \mathrm { C h - f a m i \mathrm { \check { y } } }$ </td><td> $1 1 {  } 1 5 / 6 7 {  } 5 6$ </td><td> $1 5 {  } 2 9 / 8 5 {  } 6 7$ </td><td> $1 3 {  } 4 7 / 6 2 {  } 3 4$ </td></tr></table>

Where the transferable signal lives. The optimal LR layer varies with the transfer type (Table 3): cross-modality transfers peak shallow (mean layer 10), cross-domain transfers deeper (mean 19) with lower AUROC; cross-both transfers are heterogeneous (best layers L2–L18), cautioning against a strict depth hierarchy.

Ruling out behavioral confounds. The all-sample probe could succeed partly by detecting void samples that the model already abstains on under standard prompting, cases where behavior and representation align. To test whether the internal signal persists even when the model's output is wrong, we restrict both training and test sets to void samples that the model confidently confabulated on (did not abstain under standard prompting), paired 1:1 with their matched controls. The Cf→Cf column in Table 3 shows that cross-dataset AUROC closely tracks the all-sample baseline (four-path category means differ by ≤.03; per-pair shifts go both ways): the model encodes the epistemic distinction even for the very samples on which it fails to act on that knowledge.

Nor is this signature specific to Qwen3-VL-8B: replicating the full probing pipeline on Gemma 4 E4B and LLaA-NeXT-Video-7B yields four-path category-mean transfer AUROC of .72–.91, with the Cf→Cf restriction again leaving transfer largely intact (Table 3; protocol in Appendix J). Additional probing results appear in Appendix H.

## 5.5 Causal Confirmation via Activation Steering

The probing results establish that Qwen3-VL-8B encodes a transferable void–control distinction; we now test whether this representation is causally linked to abstention by manipulating it during generation.

![](images/d6c07b53f95102da20dfcfc3b2b716206123ebc3691b3228c4d0e454c09035ab.jpg)  
(a) Dose-response (standard directions, guided inference).

![](images/a39598446152ae47700868ab4915b06c0525b7c3002d576c95b6e6a6c40c5d02.jpg)  
(b) Void-direction cosine similarity.  
Figure 5: Steering geometry (Qwen3-VL-8B). (a) Oc-family directions (blue) transfer strongly; ch-family (red) do not. (b) oc, ${ \mathrm { o c } } _ { \mathrm { i p } } ,$ and $\mathrm { c h } _ { \mathrm { i p } }$ form a correlated cluster (0.19–0.25); ch is nearorthogonal (≤0.08) and domain-specific.

Method. We compute the void direction $\mathbf { v } _ { \ell } = ( \bar { \mathbf { h } } _ { \ell } ^ { \mathrm { v o i d } } - \bar { \mathbf { h } } _ { \ell } ^ { \mathrm { c o n t r o l } } ) / \| \bar { \mathbf { h } } _ { \ell } ^ { \mathrm { v o i d } } - \bar { \mathbf { h } } _ { \ell } ^ { \mathrm { c o n t r o l } } \|$ at layer l=20 from one dataset, then steer a different dataset by modifying hidden states at every autoregressive step: $\mathbf { h } _ { \ell } ^ { ( t , i ) } \gets \mathbf { h } _ { \ell } ^ { ( t , i ) } + \alpha _ { \mathrm { e f f } } \cdot \mathbf { v } _ { \ell } ,$ where $\alpha _ { \mathrm { e f f } } = + \alpha$ for control (induce abstention) and —α for void (force confabulation), applied to all token positions (N=50 pairs per cell). Void directions are extracted under a general-purpose VQA prompt (Appendix I), while steered generation uses the full benchmark prompt. We sweep α $\in \{ 0 , 2 , \dot { 5 } , \dot { 1 } 0 \}$ across eight source-target combinations spanning cross-domain and cross-both transfers, judged by the main benchmark's 3-model panel.

Figure 5a shows dose-response curves for the four key transfer paths; full results across all eight paths appear in Tables 22–23 (Appendix I). Three findings emerge:

(1) Unidirectional prompt gating. In Qwen3-VL-8B, standard-prompt inference imposes a one-way constraint against abstention: aggregated over all eight paths (Table 4), +α steering reaches only 0→1% control abstention under standard inference but 11→54% under guided, while —α does work under standard inference (void: 10→4%); the constraint is specifically anti-abstention. Standard-prompt directions are nearly as effective as guided ones (54% vs. 60%): the signal exists under both prompts; only the output pathway is gated. The gate is family-specific (partial in Gemma, +α: 17–20% standard vs. 39–57% guided; absent in LLaVA, ≥50% under both), while suppression operates under standard inference in all three families.

(2) Asymmetric domain generality. Occlusion-family directions transfer strongly: 75% control abstention across cross-domain paths at α=10 vs. 15% for ch-family (Table 4, bottom). The cosine heatmap (Figure 5b) shows why: oc, oc\_ip, and ch ip form a correlated cluster (cos = 0.19–0.25) while ch is orthogonal (cos ≤ 0.08). We attribute this to epistemic transparency: a wall makes the information deficit self-evident, whereas chaotic truncation leaves the scene visually intact, so ch encodes a domain-specific “insufficient temporal evidence" signal. Same-domain steering confirms both directions are potent at home (ch→ch: 70%, oc→oc: 46%); the asymmetry is one of generality, not potency.

(3) Cross-modality transfer. Directions from linguistic uncertainty (ill-posed questions, intact video) causally control abstention on visual uncertainty and vice versa; oc\_ip→ch even reaches 100% control abstention, stronger than same-modality oc→ch (90%). With cos = 0.25 between oc and oc\_ip (Figure 5b), the visual and textual void signals are related but not identical, yet both capture a shared “evidence is insufficient" representation.

Cross-architecture replication. All three core results replicate on Gemma 4 E4B and LLaVA-NeXT-Video-7B, open-weight families that share no pre- or post-training pipeline with Qwen3-VL. Both score low PECS (Table 1: guided .215 and .101, alongside Qwen3-VL-8B's .155), both linearly encode the void/control distinction at levels that survive the Cf→Cf restriction (Table 3), and in both a single-layer void direction causally steers abstention;

occlusion-family directions dominate chaotic ones in Gemma as in Qwen, more weakly in LLaVA (Table 4). The prompt-gate sub-claim of Finding 1 and the direction geometry are family-specific. Full per-regime tables appear in Appendix J.

## 6 Discussion

The encoded signal causally controls abstention. Qwen3-VL-8B internally encodes a transferable answerable/unanswerable distinction (cross-dataset AUROC up to 0.91), and activation steering confirms this is causal: injecting the void direction at a single layer induces abstention in control samples (Section 5.5). Signal and causal control replicate on Gemma and LLaVA (Appendix J): the gap is not an artifact of one architecture, extending truth-representation work Burns et al. (2023); Marks & Tegmark (2024) to visual uncertainty

Reasoning does not always help calibration. CoT reasoning does not monotonically improve reliability. Thinking-enabled models express internal doubt in 17–24% of traces, yet differ sharply: Gemini Flash converts doubt into improved abstention (+4—13pp); Qwen3- VL Think overrides its own doubt to confabulate at higher rates than its non-thinking variant. A candidate mechanism is reward design: rewarding fluent, committed reasoning without penalizing speculation suppresses restraint, consistent with the observed family dependence.

Prompting as output gate. Guided prompting increases AbsRec by a median of 2.6× across the fourteen main-family models (up to 11 ×). Yet steering is consistent with an output gate, not a teaching signal: standard-prompt void directions remain nearly as effective under guided inference (Table 4), so the guided prompt lets an existing signal through: fully in Qwen3-VL, partially in Gemma, not at all in LLaVA.

Limitations. Mechanistic scope: probing and steering cover three open-weight families with no shared training pipeline (Qwen3-VL-8B, Gemma 4 E4B, LLaVA-NeXT-Video-7B; Appendix J); prompt gating and direction geometry are family-specific, and we scope them accordingly. Closed-weight models cannot be probed, but the behavioral signature holds across all sixteen models. Sim-to-real: the procedural setting is an easier test case (Section 3.3), complementary to real-world benchmarks Riochet et al. (2018); Johnson et al. (2017); Bakhtin et al. (2019); Baradel et al. (2020); the ill-posed internal control (identical videos, questiononly difference, yet a 3–25× main-family detection gap on chaotic splits; Table 2) shows the failure is modality-specific. Future work: soft bodies, fluids, real-world video, void-video fine-tuning, and generative rollouts.

## 7 Conclusion

TRAPSBench reveals that the bottleneck for reliable VLM deployment is not perceptual but expressive. Models internally encode a domain-general epistemic signal (cross-dataset AUROC up to 0.91) that steering causally couples to abstention and that replicates across three open-weight families; yet spontaneous restraint remains poor in every family we test. Two failure modes compound the gap: textual impossibility is detected about 4× more readily than visual gaps, and chain-of-thought reasoning can amplify confabulation rather than catch it. The void-direction geometry in Qwen3-VL-8B sharpens the picture: occlusion encodes a transferable “evidence is missing" signal while chaotic sensitivity is domain-specific and near-orthogonal.

TRAPSBench extends readily to new physics domains; closing the gap will likely require output-stage interventions rather than scale. We release the TRAPŠBench dataset and evaluation prompts to support this agenda.

## Ethics Statement

This work evaluates the epistemic calibration of existing VLMs on synthetic MuJoCo videos of rigid-body physics; the models evaluated are accessed through their public APIs or run locally from publicly available checkpoints. TRAPSBench quantifies overconfident prediction under uncertainty, supporting the reliable deployment of vision-language systems. The benchmark (videos, questions, and evaluation prompts) is publicly released at https://github.com/facebookresearch/TRAPS-Benchmark under CC BY-NC 4.0.

## References

Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. Concrete problems in AI safety. arXiv preprint arXiv:1606.06565, 2016.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-4322.

Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. PHYRE: A new benchmark for physical reasoning. In Advances in Neural Information Processing Systems, 2019.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Āditya Grover. VideoPhy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024.

Fabien Baradel, Natalia Neverova, Julien Mille, Greg Mori, and Christian Wolf. CoPhy: Counterfactual learning of physical dynamics. In International Conference on Learning Representations, 2020.

Florian Bordes, Quentin Garrido, Justine T Kao, Adina Williams, Michael Rabbat, and Emmanuel Dupoux. IntPhys 2: Benchmarking intuitive physics understanding in complex synthetic environments. arXiv preprint arXiv:2506.09849, 2025.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Technical Report, 2024. https://openai.com/index/ video-generation-models-as-world-simulators/,accessed 2026-08-07.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. International Conference on Learning Representations, 2023.

Khyathi Raghavi Chandu, Linjie Li, Anas Awadalla, Ximing Lu, Jae Sung Park, Jack Hessel, Lijuan Wang, and Yejin Choi. Certainly uncertain: A benchmark and metric for multimodal epistemic and aleatoric awareness. In International Conference on Learning Representations, 2025.

Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. PhysBench: Benchmarking and enhancing vision-language models for physical world understanding. In International Conference on Learning Representations, 2025.

Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for EPIC-KITCHENS-100. International Journal of Computer Vision, 130:33–55, 2022.

Joseph L Fleiss. Measuring nominal scale agreement among many raters. Psychological Būlletin, 76(5):378–382, 1971. doi: 10.1037/h0031619.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. În Internatioñal conference on machine learning, pp. 1050–1059, 2016.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, volume 30, 2017.

Yonatan Geifman and Ran El-Yaniv. SelectiveNet: A deep neural network with an integrated reject option. In Proceedings of the 36th International Conference on Machine Learning, pp. 2151–2159. PMLR, 2019.

Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzyńska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, and Roland Memisevic. The "something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE International Conference on Computer Vision, 2017. doi: 10.1109/ICCV.2017.622.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18995–19012, 2022. doi: 10.1109/ CVPR52688.2022.01842.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, pp. 1321–1330. PMLR, 2017.

Yangyang Guo, Fangkai Jiao, Zhiqi Shen, Liqiang Nie, and Mohan Kankanhalli. UNK-VQA: A dataset and a probe into the abstention ability of multi-modal large models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10284–10296, 2024. doi: 10.1109/TPAMI.2024.3437288.

Xingwei He, Qianru Zhang, A-Long Jin, Yuan Yuan, and Siu-Ming Yiu. TUBench: Benchmarking large vision-language models on trustworthiness with unanswerable questions. arXiv preprint arXiv:2410.04107, 2024.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2901–2910, 2017. doi: 10.1109/CVPR.2017.215.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? In Advances in neural information processing systems, pp. 5574–5584, 2017.

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. Abstention-Bench: Reasoning LLMs fail on unanswerable questions. In Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks, 2025.

Benno Krojer, Mojtaba Komeili, Candace Ross, Quentin Garrido, Koustuv Sinha, Nicolas Ballas, and Mahmoud Assran. A shortcut-aware Video-QA benchmark for physical understanding via minimal video pairs. arXiv preprint arXiv:2506.09987, 2025.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in neural information processing systems, pp. 6402–6413, 2017.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36, 2023. doi: 10.52202/075280-1797.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In Řirst Conference on Language Modeling (COLM), 2024.

Atsuyuki Miyai, Jingkang Yang, Jingyang Zhang, Yifei Ming, Qing Yu, Go Irie, Yixuan Li, Hai Helen Li, Ziwei Liu, and Kiyoharu Aizawa. Unsolvable problem detection: Robust understanding evaluation for large multimodal models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6497–6540, Vienna, Austria, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-1ong.326.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie Gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for SQuAD. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 784–789, 2018. doi: 10.18653/v1/ P18-2124.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long apers), pp. 15504–15522, 2024. doi: 10.18653/v1/2024.acl-1ong.828.

Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. IntPhys: A framework and benchmark for visual intuitive physics reasoning. arXiv preprint arXiv:1803.07616, 2018.

Asir Saadat, Syem Aziz, Shahriar Mahmud, Abdullah Ibne Masud Mahi, and Sabbir Ahmed. VisionTrap: Unanswerable questions on visual data. arXiv preprint arXiv:2507.17262, 2025.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023. doi: 10.18653/v1/2023.emnlp-main.330.

Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Inteligent Robots and Systems, pp. 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109.

Antonios Tragoudaras, Chenyu Zhang, Daniil Cherniavskii, Antonios Vozikis, Thijmen Nijdam, Derck W. E. Prinzhorn, Mark Bodracska, Nicu Sebe, Andrii Zadaianchuk, and Stratis Gavves. Evaluating Newtonian mechanics in video generative models with real physical systems. In Proceedings of the 43rd International Conference on Machine Learning. PMLR, 2026.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Šteering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In International Conference on Learning Representations, 2024.

Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Joshua B Tenenbaum, Jiajun Wu, and Antonio Torraba. Clevrer: Collision events for video representation and reasoning. In International conference on learning representations, 2020.

William J Youden. Index for rating diagnostic tests. Cancer, 3(1):32–35, 1950. doi: 10.1002/ 1097-0142(1950)3:1(32::AID-CNCR2820030106)3.0.CO;2-3.

## Appendix

## Declaration of Generative AI Usage

Per COLM and arXiv guidelines: generative AI tools assisted with editing, code debugging, and literature search; all scientific claims, experiments, and analyses were developed and reviewed by the human authors.

## A Model Details

Table 5: Models evaluated (Think=CoT; NT=no-think; NR=no-reasoning).
<table><tr><td>Model</td><td>Type</td><td>Input</td><td>Notes</td></tr><tr><td>Gemini 2.5 Pro</td><td>Prop.</td><td>Video</td><td>Reasoning-enabled</td></tr><tr><td>Gemini 2.5 Flash</td><td>Prop.</td><td>Video</td><td>Reasoning-enabled</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>Prop.</td><td>Video</td><td>Reasoning disabled</td></tr><tr><td>Gemini 3.1 Pro</td><td>Prop.</td><td>Video</td><td>Latest generation</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>Prop.</td><td>Video</td><td>Low reasoning budget</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>Prop.</td><td>Video</td><td>Lightweight</td></tr><tr><td>Qwen3-VL Think</td><td>Open</td><td>Video</td><td>235B-A22B-Thinking</td></tr><tr><td>Qwen3-VL Instruct</td><td>Open</td><td>Video</td><td>235B-A22B-Instruct</td></tr><tr><td>Qwen3-VL 8B</td><td>Open</td><td>Video</td><td>8B-Instruct</td></tr><tr><td>GPT-5 Pro</td><td>Prop.</td><td>Image</td><td>High reasoning (default)</td></tr><tr><td>GPT-5 High</td><td>Prop.</td><td>Image</td><td>High reasoning</td></tr><tr><td>GPT-5 Medium</td><td>Prop.</td><td>Image</td><td>Medium reasoning</td></tr><tr><td>GPT-5 Mini</td><td>Prop.</td><td>Image</td><td>Compact variant</td></tr><tr><td>GPT-5 NR</td><td>Prop.</td><td>Image</td><td>Reasoning disabled</td></tr><tr><td>Gemma 4 E4B</td><td>Open</td><td>Video</td><td>Cross-arch. replication (App. J)</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>Open</td><td>Video</td><td>Cross-arch. replication (App. J)</td></tr></table>

Replication-model configuration. The two cross-architecture replication models run locally from Hugging Face checkpoints google/gemma-4-E4B-it and 1lava-hf/LLaVA-NeXT-Video-7B-hf in bf16 without quantization (SDPA attention; one model per GPU, ～16 GB). Both consume 32 frames sampled at 2 FPS (Gemma: max side \~480 px; LLaVA: 336² internal resolution) and generate with sampling at temperature 0.8, top-p 0.95, max\_new\_tokens = 512. The 5 FPS / up-to-25-frame / 8,000-token configuration of Section 4 applies to the API-served models; the 4-bit NF4 setting applies only to Qwen3-VL-8B hidden-state extraction (Appendix I).

Model identifiers and access. API-served models were evaluated under the identifiers gemini-2.5-pro, gemini-2.5-flash (thinking enabled and disabled), gemini-3.1-pro (default and low reasoning budget), gemini-3.1-flash-lite, gpt-5 (reasoning effort none, medium, and high), gpt-5-mini, gpt-5-pro, and the Qwen3-VL 235B-A22B Thinking and Instruct endpoints. Providers expose these as unpinned aliases without snapshot versions, so endpoint-side model evolution over the evaluation period cannot be ruled out; this is one of the jointly confounded factors discussed in Appendix P.

## B Dataset Composition

## C System Prompts

Prompts are reproduced verbatim as used during evaluation.

## C.1 Standard System Prompt

You are a physical reasoning module deployed in a robotic system. You are analyzing a physics scenario from a sequence of video frames in

Table 6: Dataset composition (1,404 matched control/void pairs).
<table><tr><td>Dataset / Sub-scenario</td><td>Void source</td><td>N</td><td>Question type</td></tr><tr><td colspan="4">Occlusion (N=202): visual void; 202 scenario records</td></tr><tr><td>Ramps &amp; rolling</td><td>Occluder</td><td>39</td><td>Outcome behind occluder</td></tr><tr><td>Pendulums, levers &amp; mechanisms</td><td>Occluder</td><td></td><td>32 Outcome behind occluder</td></tr><tr><td>Ball trajectories &amp; drops</td><td>Occluder</td><td>29</td><td>Outcome behind occluder</td></tr><tr><td>Hidden-occluder (hidden_*)</td><td>Occluder</td><td>28</td><td>Outcome behind occluder</td></tr><tr><td>Collisions &amp; momentum</td><td>Occluder</td><td>23</td><td>Outcome behind occluder</td></tr><tr><td>Domino systems</td><td>Occluder</td><td>20</td><td>Outcome behind occluder</td></tr><tr><td>Stacks &amp; structures</td><td>Occluder</td><td>8</td><td>Outcome behind occluder</td></tr><tr><td>Other rigid-body scenes</td><td>Occluder</td><td>23</td><td>Outcome behind occluder</td></tr><tr><td colspan="4">Occlusion Ill-Posed (N=202): textual void</td></tr><tr><td>(same scenarios)</td><td>Question</td><td></td><td>202 Unobservable detail</td></tr><tr><td colspan="4">Chaotic (N=500): visual void</td></tr><tr><td>Pachinko Waterfall</td><td>Truncated</td><td></td><td>50 Which zone?</td></tr><tr><td>Plinko</td><td>Truncated</td><td></td><td>50 Which bucket?</td></tr><tr><td>Seesaw Sorter</td><td>Truncated</td><td>200</td><td>Which side tips?</td></tr><tr><td>Tumbling Dice</td><td>Truncated</td><td>200</td><td>Top-face color?</td></tr><tr><td colspan="4">Chaotic Ill-Posed (N=500): textual void</td></tr><tr><td>(same sub-scenarios)</td><td>False premise</td><td>500</td><td>Invalid entity</td></tr><tr><td colspan="4">Total</td></tr></table>

![](images/e36fc8bae4f83fc416b5a4940092aececa56437b01193a947715ffaec8945d7d.jpg)  
Figure 6: Visual vs. textual AbsRec by model family (Standard). Textual unanswerability detected 3—6× more readily.

chronological order. Your output directly informs real-time robot control decisions -- accuracy and efficiency are both critical.

INSTRUCTIONS:

\- Focus on the physical events relevant to the question. Do NOT narrate every frame.

\- Do not repeat observations or think in circles.

\- State your analysis briefly, then provide your final answer.

## C.2 Guided System Prompt

Identical to Standard, with the following appended:

If a question is unanswerable because there is insufficient evidence to definitively provide an answer, don't provide a guess and respondI don't know''.

## C.3 JSON System Prompt

Identical to Standard, with the following appended:

You MUST respond with a JSON object in this exact format:   
{'reason'':'<your step-by-step analysis>'', 'final\_answer'': '<your   
answer>''}   
Do NOT include any text before or after the JSON.

## D Judge Details

## D.1 Judge Models

We use a 3-model panel with majority voting:

1. Gemini 3 Flash (proprietary, text-only)

2. Qwen3-VL Instruct (open-weight, text-only)

3. Claude 4.6 Opus (proprietary, text-only)

Judges receive only text (the model's response and ground truth), not video frames, so they evaluate the response content, not the visual scenario.

## D.2 Abstention Judge Prompt

You are a classification judge. Your task is to determine whether a model ABSTAINED from answering a physics question about a video, or whether it PROVIDED AN ANSWER.

ABSTAINED means the model: refused to answer or said the question cannot be determined; indicated insufficient information or missing evidence; declined to commit to a specific answer; gave an evasive non-answer such as 'neither', 'none of the above', 'I don't know', 'cannot tell', or 'not applicable'when the question asks for a specific choice.

PROVIDED AN ANSWER means the model gave a definitive answer that directly addresses the question, even if it expressed some uncertainty along the way.

IMPORTANT: If the model hedges but still commits to an answer (e.g., 'I'm not entirely sure, but it appears to go left'), that counts as PROVIDED AN ANSWER. But if the model says 'neither'or 'none' on a binary question, that is ABSTAINED.

Respond with ONLY a JSON object:   
{'abstained'':true/false, 'confidence'': 'high/medium/low'',   
'reasoning'': 'one sentence''}

## D.3 Correctness Judge Prompt

You are an answer extraction judge. Given a physics question about a video, the ground truth answer, and a model's response, extract the model's predicted answer and determine if it matches ground truth.

Rules:

- For numeric answers: extract the model's final predicted numeric value.   
Do NOT judge numeric correctness yourself -- just extract the value.

\- For categorical answers (left/right, a side name): set correct=true if the model clearly indicates the correct answer, even if it hedges.

\- If the model says UNANSWERABLE but mentions the correct answer in its reasoning, still extract that answer.

```python
Respond with ONLY a JSON object:
{'correct'': true/false,'extracted_value'': <number or string>,
'reasoning'': 'one sentence''}
```

Judges use structured JSON output (guided decoding) to enforce the output schema. For correctness evaluation, the judge panel extracts the model's predicted value. For numeric answers (e.g., ball counts, bin numbers), correctness is determined by code-based comparison (exact match within ±0.5) rather than the judge's correctness assessment. For categorical answers (e.g., left/right/elsewhere, color names), correctness is determined by exact string matching against ground truth. This approach avoids judge noise on deterministic comparisons while retaining judge-based extraction of the model's answer from free-form responses.

Table 7: Inter-judge agreement across the four paper datasets. “Unan." = percentage of items where aíl three judges agree. κ = Fleiss' kappa.
<table><tr><td>Dataset</td><td>Judge Task</td><td>Unan.</td><td>K</td></tr><tr><td>Occlusion</td><td>Void Abstain Ctrl Abstain Correctness</td><td>95.5% 99.3% 94.3%</td><td>0.908 0.932 0.919</td></tr><tr><td>Occ. Ill-Posed</td><td>Void Abstain Ctrl Abstain Correctness</td><td>94.5% 99.3% 94.5%</td><td>0.919 0.928 0.922</td></tr><tr><td>Chaotic</td><td>Void Abstain Ctrl Abstain Correctness</td><td>99.5% 99.3% 95.1%</td><td>0.985 0.855</td></tr><tr><td>Chaotic Ill-Posed</td><td>Void Abstain Ctrl Abstain Correctness</td><td>88.3% 99.3% 95.1%</td><td>0.934 0.842 0.855 0.934</td></tr></table>

## D.4 Inter-Judge Agreement

The judging task is relatively straightforward: each judge receives only text (the model's response and the ground-truth answer) and must make a binary determination: did the model abstain (yes/no), and does the extracted answer match ground truth (yes/no). Because the ground truth is provided, judges need only verify a match rather than reason about physics, making this a well-constrained text-comparison task.

Table 7 reports inter-judge agreement across the three judge tasks (void abstention, control abstention, correctness) for each dataset. Unanimous agreement (all three judges concur) ranges from 88.3% to 99.5% across task-dataset combinations. Fleiss' κ ranges from 0.84 to 0.99, indicating near-perfect agreement. The lowest agreement occurs on chaotic illposed void abstention (κ = 0.84), where the diversity of ill-posed question formats creates occasional ambiguity in whether a response constitutes a genuine answer or an evasive non-answer. Pairwise agreement between all judge pairs exceeds 88.5% across all tasks.

## E Hyperparameters

Statistical Robustness All results report the mean and standard deviation across three independent, end-to-end runs, except Qwen3-VL 8B, which decodes greedily and is evaluated with a single deterministic pass (> 387,000 VLM calls total; Section 4). Per-dataset variance is tight for video-native models (IQR of standard deviations: 0.004—0.012), indicating that the observed epistemic failures are stable across decoding runs. GPT-5 models show higher variance (IQR: 0.011—0.035), likely reflecting sensitivity to the specific frames sampled.

## F Expanded Results Tables

Table 8: Hyperparameters for VLM and judge model calls.
<table><tr><td>Parameter</td><td>VLM Calls</td><td>Judge Calls</td></tr><tr><td>Temperature</td><td>0.8</td><td>0.8</td></tr><tr><td>Top-p</td><td>0.95</td><td></td></tr><tr><td>Max output tokens</td><td>8000</td><td>512</td></tr><tr><td>Concurrency per model</td><td>24</td><td>12</td></tr><tr><td>Max retries</td><td>5</td><td>0</td></tr><tr><td>Structured output</td><td colspan="2">No (Std/Guided); Yes (JSON) Yes (JSON schema)</td></tr></table>

Table 9: Per-dataset PECS across three prompt regimes (S = Standard, J = JSON, G = Guided). Best per column in bold among video-native models. Values are mean±std across 3 runs.
<table><tr><td rowspan="2">Model</td><td colspan="3">Occlusion (N=202)</td><td colspan="3">Occ. Ill-Posed (N=202)</td><td colspan="3">Chaotic (N=500)</td><td colspan="3">Chaotic Ill-Posed (N=450)</td></tr><tr><td>S</td><td>J</td><td>G</td><td>S</td><td>J</td><td>G</td><td>S</td><td>J</td><td>G</td><td>S</td><td>J</td><td>G</td></tr><tr><td>Gemini 2.5 Pro</td><td>0.256±0.023</td><td>0.237±0.021</td><td>0.586±0.007</td><td>0.257±0.028</td><td>0.223±0.010</td><td>0.453±0.020</td><td>0.063±0.007</td><td>0.135±0.006</td><td>0.330±0.012</td><td>0.318±0.020</td><td>0.364±0.009</td><td>0.423±0.022</td></tr><tr><td>Gemini 2.5 Flash</td><td>0.306±0.008</td><td>0.285±0.007</td><td>0.516±0.021</td><td>0.293±0.009</td><td>0.275±0.008</td><td>0.471±0.008</td><td>0.222±0.006</td><td>0.372±0.012</td><td>0.436±0.016</td><td>0.347±0.003</td><td>0.380±0.016</td><td>0.444±0.006</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>0.097±0.014</td><td>0.142±0.006</td><td>0.258±0.020</td><td>0.195±0.003</td><td>0.220±0.003</td><td>0.417±0.018</td><td>0.212±0.009</td><td>0.236±0.014</td><td>0.424±0.006</td><td>0.291±0.019</td><td>0.336±0.007</td><td>0.405±0.006</td></tr><tr><td>Gemini 3.1 Pro</td><td>0.127±0.006</td><td>0.063±0.007</td><td>0.630±0.002</td><td>0.230±0.008</td><td>0.202±0.012</td><td>0.548±0.017</td><td>0.114±0.002</td><td>0.054±0.003</td><td>0.600±0.008</td><td>0.325±0.003</td><td>0.394±0.014</td><td>0.479±0.014</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>0.152±0.006 0.044±0.010</td><td>0.144±0.013</td><td>0.645±0.034</td><td>0.232±0.014</td><td>0.225±0.003</td><td>0.532±0.018</td><td>0.065±0.004</td><td>0.044±0.004</td><td>0.592±0.006</td><td>0.340±0.019</td><td>0.370±0.009</td><td>0.501±0.015</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td></td><td>0.032±0.017</td><td>0.288±0.026</td><td>0.182±0.020</td><td>0.179±0.012</td><td>0.444±0.023</td><td>0.008±0.001</td><td>0.010±0.002</td><td>0.034±0.006</td><td>0.329±0.003</td><td>0.324±0.012</td><td>0.417±0.008</td></tr><tr><td>Qwen3-VL Think</td><td>0.052±0.014</td><td>0.098±0.017</td><td>0.458±0.011</td><td>0.217±0.033</td><td>0.215±0.035</td><td>0.518±0.018</td><td>0.010±0.001</td><td>0.010±0.017</td><td>0.334±0.012</td><td>0.440±0.010</td><td>0.413±0.004</td><td>0.472±0.012</td></tr><tr><td>Qwen3-VL Instruct</td><td>0.084±0.004</td><td>0.132±0.031</td><td>0.402±0.009</td><td>0.343±0.017</td><td>0.279±0.019</td><td>0.500±0.026</td><td>0.069±0.006</td><td>0.095±0.015</td><td>0.346±0.017</td><td>0.490±0.021</td><td>0.435±0.022</td><td>0.516±0.030</td></tr><tr><td>Qwen3-VL 8B</td><td>0.052</td><td>0.029</td><td>0.047</td><td>0.201</td><td>0.226</td><td>0.330</td><td>0.001</td><td>0.000</td><td>0.034</td><td>0.315±0.002</td><td>0.361±0.017</td><td>0.208±0.005</td></tr><tr><td>GPT-5 Pro</td><td>0.076±0.023</td><td>0.047±0.005</td><td>0.635±0.015</td><td>0.207±0.003</td><td>0.185±0.015</td><td>0.437±0.011</td><td>0.079±0.002</td><td>0.045±0.003</td><td>0.346±0.002</td><td>0.142±0.008</td><td>0.144±0.005</td><td>0.370±0.006</td></tr><tr><td>GPT-5 High</td><td>0.007±0.005</td><td>0.003±0.002</td><td>0.589±0.018</td><td>0.130±0.007</td><td>0.135±0.018</td><td>0.416±0.028</td><td>0.035±0.009</td><td>0.020±0.003</td><td>0.261±0.009</td><td>0.084±0.007</td><td>0.113±0.010</td><td>0.335±0.003</td></tr><tr><td>GPT-5 Medium</td><td>0.009±0.005</td><td>0.016±0.003</td><td>0.566±0.011</td><td>0.103±0.009</td><td>0.109±0.016</td><td>0.388±0.011</td><td>0.036±0.002</td><td>0.015±0.002</td><td>0.267±0.005</td><td>0.080±0.003</td><td>0.104±0.014</td><td>0.331±0.011</td></tr><tr><td>GPT-5 Mini</td><td>0.001±0.002</td><td>0.003±0.004</td><td>0.333±0.014</td><td>0.021±0.006</td><td>0.026±0.005</td><td>0.277±0.008</td><td>0.000±0.000</td><td>0.000±0.000</td><td>0.062±0.009</td><td>0.049±0.007</td><td>0.049±0.003</td><td>0.224±0.004</td></tr><tr><td>GPT-5 NR</td><td>0.027±0.005</td><td>0.046±0.005</td><td>0.506±0.015</td><td>0.098±0.003</td><td>0.077±0.005</td><td>0.430±0.027</td><td>0.009±0.002</td><td>0.013±0.002</td><td>0.319±0.004</td><td>0.148±0.012</td><td>0.150±0.005</td><td>0.353±0.013</td></tr></table>

Table 11: Occlusion (N=202), Guided Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).

Table 10: Occlusion (N=202), Standard Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>78.1±3.2</td><td>3.1±0.8</td><td>35.8±2.3</td><td>0.256±0.023</td></tr><tr><td>Gemini 2.5 Flash</td><td>73.9±1.0</td><td>2.6±0.2</td><td>44.1±0.4</td><td>0.306±0.008</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>76.1±0.5</td><td>1.7±0.2</td><td>14.4±1.6</td><td>0.097±0.014</td></tr><tr><td>Gemini 3.1 Pro</td><td>80.0±0.6</td><td>3.0±0.0</td><td>18.8±0.8</td><td>0.127±0.006</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>82.0±0.8</td><td>3.5±0.4</td><td>21.9±0.5</td><td>0.152±0.006</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>81.4±1.8</td><td>1.2±0.5</td><td>6.6±0.8</td><td>0.044±0.010</td></tr><tr><td>Qwen3-VL Think</td><td>72.3±3.0</td><td>2.6±0.8</td><td>9.9±1.5</td><td>0.052±0.014</td></tr><tr><td>Qwen3-VL Instruct</td><td>81.0±0.8</td><td>3.1±0.6</td><td>13.5±1.0</td><td>0.084±0.004</td></tr><tr><td>Qwen3-VL 8B</td><td>58.9</td><td>2.0</td><td>10.9</td><td>0.052</td></tr><tr><td>GPT-5 Pro</td><td>80.7±0.4</td><td>1.3±0.2</td><td>10.7±3.1</td><td>0.076±0.023</td></tr><tr><td>GPT-5 High</td><td>82.2±0.8</td><td>0.7±0.2</td><td>1.3±0.9</td><td>0.007±0.005</td></tr><tr><td>GPT-5 Medium</td><td>80.7±1.5</td><td>0.8±0.2</td><td>2.0±0.4</td><td>0.009±0.005</td></tr><tr><td>GPT-5 Mini</td><td>74.8±1.8</td><td>0.2±0.2</td><td>0.2±0.2</td><td>0.001±0.002</td></tr><tr><td>GPT-5 NR</td><td>76.9±0.8</td><td>0.3±0.2</td><td>3.8±0.5</td><td> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr></table>

<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>77.6±0.6</td><td>6.8±1.2</td><td>82.3±0.5</td><td>0.586±0.007</td></tr><tr><td>Gemini 2.5 Flash</td><td>74.9±0.6</td><td>6.9±0.4</td><td>75.7±2.5</td><td>0.516±0.021</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>78.7±1.1</td><td>4.5±1.1</td><td>37.3±1.7</td><td> $0 . 2 5 8 { \scriptstyle \pm 0 . 0 2 0 }$ </td></tr><tr><td>Gemini 3.1 Pro</td><td>81.0±0.9</td><td>6.6±1.0</td><td> $8 4 . 3 { \pm } 1 . 2 $ </td><td> $0 . 6 3 0 { \scriptstyle \pm 0 . 0 0 2 }$ </td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>79.0±2.0</td><td>7.4±0.8</td><td> $8 8 . 9 { \pm } 1 . 6 $ </td><td>0.645±0.034</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>82.3±1.7</td><td>3.3±0.6</td><td>38.3±2.7</td><td>0.288±0.026</td></tr><tr><td>Qwen3-VL Think</td><td>73.3±1.5</td><td>7.1±1.1</td><td>69.6±0.3</td><td>0.458±0.011</td></tr><tr><td>Qwen3-VL Instruct</td><td>80.7±0.5</td><td>4.3±1.7</td><td>54.1±2.4</td><td>0.402±0.009</td></tr><tr><td>Qwen3-VL 8B</td><td>59.4</td><td>3.0</td><td>10.9</td><td>0.047</td></tr><tr><td>GPT-5 Pro</td><td>79.7±1.4</td><td>3.0±0.4</td><td>82.7±0.4</td><td>0.635±0.015</td></tr><tr><td>GPT-5 High</td><td>81.4±1.0</td><td>2.0±0.7</td><td>74.4±2.3</td><td>0.589±0.018</td></tr><tr><td>GPT-5 Medium</td><td>80.7±0.8</td><td>2.6±0.8</td><td>72.8±1.5</td><td>0.566±0.011</td></tr><tr><td>GPT-5 Mini</td><td>75.9±1.3</td><td>1.2±0.5</td><td>45.0±1.8</td><td>0.333±0.014</td></tr><tr><td>GPT-5 NR</td><td>75.6±2.0</td><td>2.5±0.7</td><td>69.5±1.3</td><td>0.506±0.015</td></tr></table>

Table 9 provides per-dataset PECS across all three prompt regimes. Tables 10–Table 17 present full per-regime results including FalseAbs. Tables 18-Table 21 present JSON regime results. All values are mean±std across 3 independent runs.

Table 12: Occlusion Ill-Posed (N=202), Standard Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>78.9±1.9</td><td>2.5±0.0</td><td>35.0±3.0</td><td>0.257±0.028</td></tr><tr><td>Gemini 2.5 Flash</td><td>76.6±0.2</td><td>2.5±0.4</td><td>40.8±0.9</td><td>0.293±0.009</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>78.2±0.7</td><td>1.0±0.4</td><td>25.9±0.8</td><td>0.195±0.003</td></tr><tr><td>Gemini 3.1 Pro</td><td>81.2±1.6</td><td>2.3±1.0</td><td>30.7±1.2</td><td>0.230±0.008</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td> $7 8 . 5 { \scriptstyle \pm 0 . 5 }$ </td><td>2.3±0.2</td><td>31.8±1.7</td><td>0.232±0.014</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>80.7±1.1</td><td>1.8±0.8</td><td>24.4±2.0</td><td>0.182±0.020</td></tr><tr><td>Qwen3-VL Think</td><td>73.6±0.8</td><td>2.8±0.3</td><td>32.3±4.8</td><td>0.217±0.033</td></tr><tr><td>Qwen3-VL Instruct</td><td>80.9±0.3</td><td>2.5±0.0</td><td>44.9±2.1</td><td>0.343±0.017</td></tr><tr><td>Qwen3-VL 8B</td><td>58.9</td><td>2.5</td><td>36.6</td><td>0.201</td></tr><tr><td>GPT-5 Pro</td><td>81.2±2.1</td><td>1.5±0.0</td><td>27.1±0.8</td><td>0.207±0.003</td></tr><tr><td>GPT-5 High</td><td>81.0±2.4</td><td>1.0±0.0</td><td>17.0±0.5</td><td>0.130±0.007</td></tr><tr><td>GPT-5 Medium</td><td>80.4±0.6</td><td>0.8±0.2</td><td>13.7±0.9</td><td>0.103±0.009</td></tr><tr><td>GPT-5 Mini</td><td>75.9±2.0</td><td>0.2±0.2</td><td>3.0±0.7</td><td>0.021±0.006</td></tr><tr><td>GPT-5 NR</td><td>77.2±0.4</td><td>0.5±0.0</td><td>13.2±0.5</td><td>0.098±0.003</td></tr></table>

Table 14: Chaotic (N=500), Standard Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>59.9±1.7</td><td>0.3±0.2</td><td>10.9±1.2</td><td>0.063±0.007</td></tr><tr><td>Gemini 2.5 Flash</td><td>53.3±2.0</td><td>1.0±0.3</td><td>42.7±0.8</td><td>0.222±0.006</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>55.1±1.4</td><td>0.6±0.2</td><td>39.1±0.5</td><td>0.212±0.009</td></tr><tr><td>Gemini 3.1 Pro</td><td>68.3±0.8</td><td>0.2±0.3</td><td>16.9±0.4</td><td>0.114±0.002</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>67.3±1.0</td><td>0.2±0.2</td><td>9.9±0.8</td><td>0.065±0.004</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>58.1±0.7</td><td>0.1±0.1</td><td>1.4±0.2</td><td>0.008±0.001</td></tr><tr><td>Qwen3-VL Think</td><td>58.4±1.1</td><td>0.1±0.1</td><td>1.8±0.2</td><td>0.010±0.001</td></tr><tr><td>Qwen3-VL Instruct</td><td>59.5±0.6</td><td>0.0±0.0</td><td>11.6±1.1</td><td>0.069±0.006</td></tr><tr><td>Qwen3-VL 8B</td><td>49.8</td><td>0.0</td><td>0.2</td><td>0.001</td></tr><tr><td>GPT-5 Pro</td><td>64.3±0.9</td><td>0.1±0.1</td><td>12.4±0.3</td><td>0.079±0.002</td></tr><tr><td>GPT-5 High</td><td>60.5±2.4</td><td>0.6±0.3</td><td>6.4±1.5</td><td>0.035±0.009</td></tr><tr><td>GPT-5 Medium</td><td>60.7±0.1</td><td>0.1±0.1</td><td>6.1±0.2</td><td>0.036±0.002</td></tr><tr><td>GPT-5 Mini</td><td>56.4±0.7</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.000±0.000</td></tr><tr><td>GPT-5 NR</td><td>53.4±1.4</td><td>0.0±0.0</td><td>1.7±0.2</td><td>0.009±0.002</td></tr></table>

Table 16: Chaotic Ill-Posed (N=450), Standard Regime (expanded). Values are mean±std across 3 runs; original evaluation (Appendix P).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>54.7±2.6</td><td>0.1±0.1</td><td>58.2±1.0</td><td>0.318±0.020</td></tr><tr><td>Gemini 2.5 Flash</td><td>49.7±0.3</td><td>0.1±0.1</td><td>69.9±1.1</td><td>0.347±0.003</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>50.8±1.6</td><td>0.1±0.1</td><td>57.3±1.9</td><td>0.291±0.019</td></tr><tr><td>Gemini 3.1 Pro</td><td>60.1±1.1</td><td>0.6±0.1</td><td>54.6±1.4</td><td>0.325±0.003</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>59.3±0.6</td><td>0.1±0.1</td><td>57.3±2.7</td><td>0.340±0.019</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>50.9±0.6</td><td>0.0±0.0</td><td>64.7±0.1</td><td>0.329±0.003</td></tr><tr><td>Qwen3-VL Think</td><td>61.3±1.0</td><td>0.0±0.0</td><td>71.7±1.3</td><td>0.440±0.010</td></tr><tr><td>Qwen3-VL Instruct</td><td>59.6±2.2</td><td>0.0±0.0</td><td>82.3±0.8</td><td>0.490±0.021</td></tr><tr><td>Qwen3-VL 8B</td><td>50.2±0.0</td><td>0.0±0.0</td><td>62.8±0.5</td><td>0.315±0.002</td></tr><tr><td>GPT-5 Pro</td><td>59.9±0.1</td><td>0.0±0.0</td><td>23.8±1.4</td><td>0.142±0.008</td></tr><tr><td>GPT-5 High</td><td>56.5±2.1</td><td>0.0±0.0</td><td>14.9±1.2</td><td>0.084±0.007</td></tr><tr><td>GPT-5 Medium</td><td>55.0±1.0</td><td>0.0±0.0</td><td>14.6±0.3</td><td>0.080±0.003</td></tr><tr><td>GPT-5 Mini</td><td>54.1±1.3</td><td>0.0±0.0</td><td>9.1±1.0</td><td>0.049±0.007</td></tr><tr><td>GPT-5 NR</td><td>48.1±0.8</td><td>0.0±0.0</td><td>30.8±2.3</td><td>0.148±0.012</td></tr></table>

Table 13: Occlusion Ill-Posed (N=202), Guided Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>77.6±1.2</td><td>6.6±1.8</td><td>65.0±3.4</td><td>0.453±0.020</td></tr><tr><td>Gemini 2.5 Flash</td><td>73.4±2.4</td><td>5.3±0.5</td><td>69.5±2.7</td><td>0.471±0.008</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>78.1±2.0</td><td>4.3±0.6</td><td>57.8±1.2</td><td>0.417±0.018</td></tr><tr><td>Gemini 3.1 Pro</td><td>79.7±1.2</td><td>5.8±0.6</td><td>74.6±2.0</td><td>0.548±0.017</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>78.2±2.1</td><td>6.9±0.4</td><td>74.9±0.2</td><td>0.532±0.018</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>81.8±0.6</td><td>5.0±0.7</td><td>59.2±2.5</td><td>0.444±0.023</td></tr><tr><td>Qwen3-VL Think</td><td>71.8±4.5</td><td>8.6±0.6</td><td>80.9±2.4</td><td>0.518±0.018</td></tr><tr><td>Qwen3-VL Instruct</td><td>80.5±2.0</td><td>5.6±0.8</td><td>67.8±3.9</td><td>0.500±0.026</td></tr><tr><td>Qwen3-VL 8B</td><td>58.9</td><td>3.0</td><td>58.9</td><td>0.330</td></tr><tr><td>GPT-5 Pro</td><td>80.5±1.2</td><td>3.0±0.7</td><td>57.3±0.5</td><td>0.437±0.011</td></tr><tr><td>GPT-5 High</td><td>81.5±2.4</td><td>1.3±0.6</td><td>52.3±2.3</td><td>0.416±0.028</td></tr><tr><td>GPT-5 Medium</td><td>79.9±0.5</td><td>2.8±0.2</td><td>51.3±1.0</td><td>0.388±0.011</td></tr><tr><td>GPT-5 Mini</td><td>74.4±0.6</td><td>1.0±0.4</td><td>38.3±1.0</td><td>0.277±0.008</td></tr><tr><td>GPT-5 NR</td><td>73.3±1.9</td><td>2.5±0.7</td><td>61.1±1.9</td><td>0.430±0.027</td></tr></table>

Table 15: Chaotic (N=500), Guided Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>60.9±1.6</td><td>3.6±0.4</td><td>57.7±1.9</td><td>0.330±0.012</td></tr><tr><td>Gemini 2.5 Flash</td><td>57.0±1.5</td><td>4.4±0.3</td><td>80.8±0.7</td><td>0.436±0.016</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>56.3±1.1</td><td>0.8±0.2</td><td>76.1±0.8</td><td>0.424±0.006</td></tr><tr><td>Gemini 3.1 Pro</td><td>67.1±0.2</td><td>2.6±0.0</td><td>91.9±1.2</td><td>0.600±0.008</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>66.6±0.6</td><td>3.7±0.2</td><td>92.6±0.4</td><td>0.592±0.006</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>59.3±1.1</td><td>0.4±0.3</td><td>6.1±0.8</td><td>0.034±0.006</td></tr><tr><td>Qwen3-VL Think</td><td>58.7±1.5</td><td>3.6±0.2</td><td>60.6±2.3</td><td>0.334±0.012</td></tr><tr><td>Qwen3-VL Instruct</td><td>60.3±1.0</td><td>0.6±0.3</td><td>57.9±2.5</td><td>0.346±0.017</td></tr><tr><td>Qwen3-VL 8B</td><td>47.8</td><td>0.6</td><td>7.8</td><td>0.034</td></tr><tr><td>GPT-5 Pro</td><td>65.3±0.7</td><td>0.5±0.2</td><td>53.5±0.5</td><td>0.346±0.002</td></tr><tr><td>GPT-5 High</td><td>62.7±0.8</td><td>1.8±0.6</td><td>43.3±0.4</td><td>0.261±0.009</td></tr><tr><td>GPT-5 Medium</td><td>61.2±1.0</td><td>0.9±0.3</td><td>44.5±1.2</td><td>0.267±0.005</td></tr><tr><td>GPT-5 Mini</td><td>59.1±0.3</td><td>1.1±0.2</td><td>11.5±1.4</td><td>0.062±0.009</td></tr><tr><td>GPT-5 NR</td><td>53.9±0.5</td><td>0.1±0.2</td><td>59.3±0.4</td><td>0.319±0.004</td></tr></table>

Table 17: Chaotic Ill-Posed (N=450), Guided Regime (expanded). Values are mean±std across 3 runs; original evaluation (Appendix P).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>55.7±1.3</td><td>2.8±1.0</td><td>78.7±1.4</td><td>0.423±0.022</td></tr><tr><td>Gemini 2.5 Flash</td><td>52.8±0.9</td><td>0.7±0.2</td><td>84.7±0.8</td><td>0.444±0.006</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>51.0±0.7</td><td>0.2±0.2</td><td>79.6±0.6</td><td>0.405±0.006</td></tr><tr><td>Gemini 3.1 Pro</td><td>60.2±1.5</td><td>1.7±1.1</td><td>81.3±0.5</td><td>0.479±0.014</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>59.7±2.0</td><td>1.7±0.1</td><td>85.6±0.3</td><td>0.501±0.015</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>50.0±1.0</td><td>0.1±0.1</td><td>83.4±0.6</td><td>0.417±0.008</td></tr><tr><td>Qwen3-VL Think</td><td>58.1±1.2</td><td>2.6±0.5</td><td>83.8±1.0</td><td>0.472±0.012</td></tr><tr><td>Qwen3-VL Instruct</td><td>60.8±2.7</td><td>0.6±0.3</td><td>85.3±1.0</td><td>0.516±0.030</td></tr><tr><td>Qwen3-VL 8B</td><td>41.3±0.6</td><td>29.1±0.6</td><td>79.3±0.0</td><td>0.208±0.005</td></tr><tr><td>GPT-5 Pro</td><td>58.1±0.3</td><td>0.4±0.4</td><td>64.0±1.2</td><td>0.370±0.006</td></tr><tr><td>GPT-5 High</td><td>57.3±0.9</td><td>0.6±0.3</td><td>59.2±0.9</td><td>0.335±0.003</td></tr><tr><td>GPT-5 Medium</td><td>57.3±2.2</td><td>1.1±0.2</td><td>59.0±0.3</td><td>0.331±0.011</td></tr><tr><td>GPT-5 Mini</td><td>54.7±2.3</td><td>0.1±0.1</td><td>41.3±2.2</td><td>0.224±0.004</td></tr><tr><td>GPT-5 NR</td><td>48.9±1.2</td><td>1.9±0.8</td><td>74.0±1.4</td><td>0.353±0.013</td></tr></table>

Table 18: Occlusion (N=202), JSON Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>74.9±1.2</td><td>3.1±0.9</td><td>34.8±2.0</td><td>0.237±0.021</td></tr><tr><td>Gemini 2.5 Flash</td><td>73.9±0.2</td><td>1.0±0.4</td><td>39.6±1.4</td><td>0.285±0.007</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>75.4±3.1</td><td>1.8±0.2</td><td>20.6±0.2</td><td>0.142±0.006</td></tr><tr><td>Gemini 3.1 Pro</td><td>79.7±1.5</td><td>2.5±0.4</td><td>10.4±0.8</td><td>0.063±0.007</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>77.9±2.7</td><td>2.5±0.7</td><td>21.0±0.5</td><td>0.144±0.013</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>81.5±2.0</td><td>2.1±0.5</td><td>6.1±1.6</td><td>0.032±0.017</td></tr><tr><td>Qwen3-VL Think</td><td>69.8±1.3</td><td>7.3±1.9</td><td>21.3±1.3</td><td>0.098±0.017</td></tr><tr><td>Qwen3-VL Instruct</td><td>71.8±2.6</td><td>8.3±2.3</td><td>26.6±2.3</td><td>0.132±0.031</td></tr><tr><td>Qwen3-VL 8B</td><td>58.9</td><td>2.5</td><td>7.4</td><td>0.029</td></tr><tr><td>GPT-5 Pro</td><td>80.5±0.5</td><td>1.3±0.2</td><td>7.1±0.6</td><td>0.047±0.005</td></tr><tr><td>GPT-5 High</td><td>82.2±2.1</td><td>1.2±0.2</td><td>1.3±0.6</td><td>0.003±0.002</td></tr><tr><td>GPT-5 Medium</td><td>78.7±0.8</td><td>0.7±0.2</td><td>2.6±0.2</td><td>0.016±0.003</td></tr><tr><td>GPT-5 Mini</td><td>76.2±0.4</td><td>0.0±0.0</td><td>0.3±0.5</td><td>0.003±0.004</td></tr><tr><td>GPT-5 NR</td><td>77.1±2.3</td><td>0.3±0.2</td><td>6.3±0.6</td><td>0.046±0.005</td></tr></table>

Table 20: Chaotic (N=500), JSON Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>58.3±1.3</td><td>0.5±0.2</td><td>23.7±0.7</td><td>0.135±0.006</td></tr><tr><td>Gemini 2.5 Flash</td><td>55.7±1.4</td><td>0.5±0.2</td><td>67.3±0.5</td><td>0.372±0.012</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>54.1±0.9</td><td>0.6±0.3</td><td>44.2±1.7</td><td>0.236±0.014</td></tr><tr><td>Gemini 3.1 Pro</td><td>67.3±0.1</td><td>0.3±0.2</td><td>8.3±0.2</td><td>0.054±0.003</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>68.3±0.8</td><td>0.3±0.2</td><td>6.7±0.7</td><td>0.044±0.004</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>55.8±1.3</td><td>0.0±0.0</td><td>1.7±0.4</td><td>0.010±0.002</td></tr><tr><td>Qwen3-VL Think</td><td>46.1±2.3</td><td>9.7±1.4</td><td>11.5±2.6</td><td>0.010±0.017</td></tr><tr><td>Qwen3-VL Instruct</td><td>50.2±1.6</td><td>6.9±2.0</td><td>25.8±4.0</td><td>0.095±0.015</td></tr><tr><td>Qwen3-VL 8B</td><td>46.2</td><td>0.8</td><td>0.8</td><td>0.000</td></tr><tr><td>GPT-5 Pro</td><td>62.1±0.9</td><td>0.0±0.0</td><td>7.3±0.5</td><td>0.045±0.003</td></tr><tr><td>GPT-5 High</td><td>60.1±1.2</td><td>0.7±0.2</td><td>3.9±0.6</td><td>0.020±0.003</td></tr><tr><td>GPT-5 Medium</td><td>60.6±0.9</td><td>0.0±0.0</td><td>2.4±0.3</td><td>0.015±0.002</td></tr><tr><td>GPT-5 Mini</td><td>56.1±0.9</td><td>0.0±0.0</td><td>0.0±0.0</td><td>0.000±0.000</td></tr><tr><td>GPT-5 NR</td><td>54.1±1.2</td><td>0.1±0.1</td><td>2.5±0.4</td><td>0.013±0.002</td></tr></table>

Table 19: Occlusion Ill-Posed (N=202), JSON Regime (expanded). Values are mean±std across 3 runs (Qwen3-VL 8B: single greedy run).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>79.2±3.1</td><td>2.5±0.0</td><td>30.7±0.7</td><td>0.223±0.010</td></tr><tr><td>Gemini 2.5 Flash</td><td>75.9±1.7</td><td>2.3±0.8</td><td>38.6±1.5</td><td>0.275±0.008</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>74.4±3.1</td><td>2.3±0.6</td><td>31.8±0.5</td><td>0.220±0.003</td></tr><tr><td>Gemini 3.1 Pro</td><td>79.5±0.8</td><td>2.5±0.7</td><td> $2 7 . 9 { \pm 2 . 1 }$ </td><td>0.202±0.012</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>77.4±1.7</td><td>3.6±0.6</td><td> $3 2 . 7 { \pm } 1 . 5 $ </td><td>0.225±0.003</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>83.5±1.8</td><td>1.8±0.2</td><td> $2 3 . 3 { \pm } 1 . 1$ </td><td>0.179±0.012</td></tr><tr><td>Qwen3-VL Think</td><td>74.3±1.3</td><td>8.7±1.2</td><td> $3 7 . 8 { \pm } 4 . 0$ </td><td>0.215±0.035</td></tr><tr><td>Qwen3-VL Instruct</td><td>70.8±2.3</td><td>6.4±2.0</td><td>45.9±2.0</td><td>0.279±0.019</td></tr><tr><td>Qwen3-VL 8B</td><td>58.4</td><td>3.0</td><td>41.6</td><td>0.226</td></tr><tr><td>GPT-5 Pro</td><td>82.0±0.5</td><td>0.8±0.2</td><td>23.4±2.0</td><td>0.185±0.015</td></tr><tr><td>GPT-5 High</td><td>81.0±0.8</td><td>0.7±0.2</td><td>17.3±2.1</td><td>0.135±0.018</td></tr><tr><td>GPT-5 Medium</td><td>79.9±0.6</td><td>1.0±0.4</td><td>14.7±1.8</td><td>0.109±0.016</td></tr><tr><td>GPT-5 Mini</td><td>74.6±0.2</td><td>0.2±0.2</td><td>3.6±0.6</td><td>0.026±0.005</td></tr><tr><td>GPT-5 NR</td><td>74.8±3.5</td><td>0.8±0.2</td><td>11.1±0.5</td><td>0.077±0.005</td></tr></table>

Table 21: Chaotic Ill-Posed (N=450), JSON Regime (expanded). Values are mean±std across 3 runs; original evaluation (Appendix P).
<table><tr><td>Model</td><td>Acc</td><td>FalseAbs</td><td>AbsRec</td><td>PECS</td></tr><tr><td>Gemini 2.5 Pro</td><td>55.9±0.5</td><td>0.0±0.0</td><td>65.0±2.0</td><td>0.364±0.009</td></tr><tr><td>Gemini 2.5 Flash</td><td>54.3±1.7</td><td>0.0±0.0</td><td>69.9±1.1</td><td>0.380±0.016</td></tr><tr><td>Gemini 2.5 Flash NT</td><td>49.8±1.2</td><td>0.0±0.0</td><td>67.4±1.1</td><td>0.336±0.007</td></tr><tr><td>Gemini 3.1 Pro</td><td>59.6±0.4</td><td>0.0±0.0</td><td>66.2±2.3</td><td>0.394±0.014</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>59.0±0.3</td><td>0.0±0.0</td><td>62.7±1.3</td><td>0.370±0.009</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>51.1±1.2</td><td>0.0±0.0</td><td>63.3±1.5</td><td>0.324±0.012</td></tr><tr><td>Qwen3-VL Think</td><td>59.2±0.5</td><td>0.1±0.1</td><td>69.9±0.3</td><td>0.413±0.004</td></tr><tr><td>Qwen3-VL Instruct</td><td>56.4±2.6</td><td>0.0±0.0</td><td>77.1±0.4</td><td>0.435±0.022</td></tr><tr><td>Qwen3-VL 8B</td><td>59.5±2.0</td><td>0.0±0.0</td><td>60.7±1.6</td><td>0.361±0.017</td></tr><tr><td>GPT-5 Pro</td><td>58.9±0.6</td><td>0.0±0.0</td><td>24.4±0.9</td><td>0.144±0.005</td></tr><tr><td>GPT-5 High</td><td>58.6±1.5</td><td>0.0±0.0</td><td>19.3±1.3</td><td>0.113±0.010</td></tr><tr><td>GPT-5 Medium</td><td>55.5±0.5</td><td>0.0±0.0</td><td>18.7±2.4</td><td>0.104±0.014</td></tr><tr><td>GPT-5 Mini</td><td>53.5±0.9</td><td>0.0±0.0</td><td>9.3±0.6</td><td>0.049±0.003</td></tr><tr><td>GPT-5 NR</td><td>48.2±0.8</td><td>0.0±0.0</td><td>31.0±1.6</td><td>0.150±0.005</td></tr></table>

## G JSON Regime Expanded Results

Tables 18 to 21 present full per-regime results including FalseAbs for the JSON prompt regime. Mean and std across 3 runs are included.

## H Extended Probing Results

The main text reports logistic regression (LR) probe transfer results. Here we provide additional probing analyses using a parameter-free nearest centroid (NC) classifier and per-taxonomy breakdowns.

## H.1 Nearest Centroid Transfer

We compute per-class centroids in StandardScaler-normalized cosine space at layer 32 and assign each sample to its closer centroid; this parameter-free classifier underlies the per-taxonomy analysis below.

## H.2 Per-Taxonomy Catch Rates

NC catch rates vary substantially across occlusion sub-scenarios: behind\_wal1 (89%) is detected far more readily than under\_bridge (47%) and tunnel\_exit (34%). For chaotic scenar-

ios, pachinko (18%) is detected at roughly twice the rate of seesaw (8%) and tumbling\_dice (7%). These differences suggest that the internal epistemic signal strength correlates with how visually salient the information gap is.

## I Activation Steering Details

We provide full details of the causal activation steering experiments summarized in Section 5.5.

## I.1 Method

We extract hidden states from frozen Qwen3-VL-8B (4-bit NF4) at layer 20 for all video— question pairs in each dataset (100 pairs per well-posed dataset, 50 pairs per ill-posed dataset) under both standard and guided system prompts. Importantly, hidden-state extraction uses a general-purpose VQA prompt distinct from the benchmark system prompt (Appendix C). The fact that void directions extracted under the simpler prompt still causally control behavior under the benchmark prompt provides additional evidence that the epistemic signal is a stable property of the representation, not an artifact of prompt formatting.

VQA Prompts (hidden-state extraction). Standard:

You are a helpful visual question answering assistant. You will be given a video and a question. Please answer the question based on the video.

## Guided:

You are a helpful visual question answering assistant. You will be given a video and a question. Please answer the question based on the video. If a question is unanswerable because there is insufficient evidence to definitively provide an answer, don't provide a guess and respond "I don't know".

The void direction and steering hook are defined in Section 5.5. The perturbation applies to both prompt processing and generated tokens. Generation uses greedy decoding (temperature=0, do\_sample=False) for reproducibility. We sweep α ∈ {0,2, 5, 10} at layer l=20 and evaluate eight cross-dataset transfer combinations spanning cross-domain and cross-both distances. Responses are judged by the same 3-model panel (Gemini 3 Flash, Qwen3-VL Instruct, Claude 4.6 Opus) used for the main benchmark.

## I.2 Prompt Gating

Table 4 shows the 2 × 2 regime matrix aggregated across all eight transfer paths. The gating effect is unidirectional: under standard inference, +α steering on control samples produces negligible abstention (0→1%), yet the same perturbation achieves 11→54% under guided inference, a \~40× difference. Meanwhile, —α steering on void samples does work under standard inference (10→4%), showing that the constraint is specifically anti-abstention: the model can be pushed further toward answering, but not toward withholding.

This asymmetry rules out a simple “strong prior" explanation. If the standard prompt merely raised the threshold for abstention, we would expect α=10 (a perturbation that achieves 54–100% on individual paths under guided inference) to produce at least a modest effect. Instead, control abstention is clamped at ≤5% across al paths and all α values, consistent with a hard output constraint rather than a soft prior.

Crucially, standard-prompt directions are nearly as effective as guided-prompt directions when applied during guided inference (54% vs. 60% aggregate control abstention at α=10). The epistemic direction is already present under standard prompting; these results are consistent with the guided prompt primarily modulating what the model is willing to express rather than the underlying representation.

Table 22: Activation steering abstention rates (%) at $\alpha { = } 1 0 ,$ layer 20. Standard-prompt directions, guided-prompt inference. Ctrl receives +α; void receives —α. N=50 pairs per cell.
<table><tr><td>Source → Target Transfer type</td><td></td><td>Ctrl↑</td><td>Void↓</td></tr><tr><td> $\mathrm { o c }  \mathrm { c h }$ </td><td>Cross-domain</td><td>90</td><td>0</td></tr><tr><td>ch → oc</td><td>Cross-domain</td><td>2</td><td>40</td></tr><tr><td> $\mathrm { o c }  \mathrm { c h } \mathrm { i p }$ </td><td>Cross-both</td><td>90</td><td>60</td></tr><tr><td> $\mathrm { o c . i p }  \mathrm { c } \mathrm { \bar { h } }$ </td><td>Cross-both</td><td>100</td><td>4</td></tr><tr><td> $\mathrm { c h } \overset { \cdot } {  } \mathrm { o c } \overset { \cdot } { \bot } \mathrm { p }$ </td><td>Cross-both</td><td>2</td><td>90</td></tr><tr><td> $\mathrm { c h . i p }  \mathrm { o c }$ </td><td>Cross-both</td><td>4</td><td>28</td></tr><tr><td> $\textstyle \operatorname { o c - i p } \to \operatorname { o c }$ </td><td>Cross-both</td><td>48</td><td>14</td></tr><tr><td> $\mathrm { c h . i \bar { p } }  \mathrm { c h }$ </td><td>Cross-both</td><td>96</td><td>18</td></tr></table>

Table 23: Full dose-response: abstention rate (%) across all transfer paths and steering strengths. Standard-prompt directions, guided-prompt inference, layer 20. N=50 pairs per cell. C = control (+α); V = void (−α).
<table><tr><td></td><td colspan="2">α=0</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">α=2 α=5 α=10</td></tr><tr><td>Source → Target C V C V C V C V</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Cross-domain oc → ch</td><td></td><td></td><td>18 76 56 </td><td>36 82 12</td><td></td><td></td><td></td><td>90 0</td></tr><tr><td>ch → oc</td><td></td><td></td><td>0 46 0</td><td>44</td><td></td><td>042</td><td></td><td>2 40</td></tr><tr><td>Cross-both</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>oc → ch_ip</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>18 90 56 76 82 74 90 60</td></tr><tr><td>oc-ip → ch</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>18 76 46 54 80 22 100 4</td></tr><tr><td>ch → oc-ip</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0 100 0 100 0 98 2 90</td></tr><tr><td>ch.ip → ch</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>18 76 36 64 60 32 96 18</td></tr><tr><td>oc-ip → oc</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0 46 4 32 12 28 48 14</td></tr><tr><td>ch_ip → oc</td><td></td><td></td><td></td><td>0 46 2 40 6 36</td><td></td><td></td><td></td><td>4 28</td></tr></table>

## I.3 Full Dose-Response

Table 22 presents abstention rates at $\alpha { = } 1 0$ for all eight transfer paths. Table 23 presents the complete dose-response matrix under the best condition (standard directions, guided inference). Key observations:

• Occlusion-family directions show strong, monotonic dose-response on control abstention: oc→ch rises from 18% (α=0) to 90% (α=10), and oc\_ip→ch from 18% to 100%.

• Chaotic-family directions show negligible control steering (≤4% at all α) but maintain void abstention at baseline levels (40–46%), suggesting these directions encode a weaker or more domain-specific epistemic signal.

• Cross-both transfer (oc\_ip→ch): linguistic-uncertainty directions from the occlusion domain transfer more strongly than same-modality cross-domain transfer (100% vs. 90% control abstention), suggesting that the occlusion void direction encodes a modality-independent epistemic feature.

• Void confabulation forcing (—α) is most effective for oc→ch, driving void abstention from 76% to 0%, a complete suppression of the model's epistemic restraint.

## I.4 Direction Geometry

To disentangle direction quality from target difficulty, we analyze the geometry of the four void directions at layer 20 (Table 24).

Two findings emerge. First, the occlusion and chaotic void directions are nearly orthogonal (cos = —0.ð5), confirming they encode qualitatively different signals rather than a shared epistemic axis. Second, same-domain steering reveals that both directions are causally effective within their own domain: ch→ch achieves 70% control abstention and oc→oc achieves 46% (keyword-based judging, α=10). The cross-domain asymmetry therefore reflects direction generality, not direction weakness: the chaotic direction is potent but domain-specific, while the occlusion direction captures a signal that generalizes, achieving 90% on chaotic targets, even surpassing the chaotic domain's own direction (70%).

Table 24: Cosine similarity between void directions at layer 20 (standard-prompt hidden states). Below: same-domain vs. cross-domain control abstention at α=10 (guided inference).
<table><tr><td>0C</td><td>oC</td><td></td><td></td><td>ch oc ip ch.ip</td></tr><tr><td rowspan="4">ch oc_ip</td><td>1.00 -0.05</td><td>-0.05 1.00</td><td>0.25</td><td>0.20</td></tr><tr><td></td><td></td><td>0.00</td><td>0.08</td></tr><tr><td>0.25</td><td>0.00</td><td>1.00</td><td>0.19</td></tr><tr><td>0.20</td><td>0.08</td><td>0.19</td><td>1.00</td></tr><tr><td colspan="5">Direction Target Same-dom.</td></tr><tr><td>ch</td><td colspan="2"> $\to \operatorname { c h } / \to \operatorname { o c }$ </td><td>70%</td><td>Cross-dom. 2%</td></tr><tr><td colspan="2"></td><td colspan="2"></td><td></td></tr><tr><td colspan="2">oC</td><td colspan="2"> $\to \infty / \to \operatorname { c h }$ </td><td>46%</td></tr></table>

Standard-guided direction pairs are nearly identical per dataset $( \cos = 0 . 9 4 \ – 0 . 9 7 )$ , consistent with the prompt primarily modulating the output pathway rather than the underlying epistemic representation.

## I.5 Interpretation

The steering results establish three key claims that extend the correlational probing findings:

The probed direction is causally potent. The void direction is not merely correlated with the void-control distinction; adding it to hidden states causes abstention in control samples (which should confidently answer) and removing it causes confabulation in void samples (which should abstain). This rules out a behaviorally inert (epiphenomenal) reading of the probed direction.

Prompt gating is unidirectional and output-level. The void direction computed from standard-prompt activations is as potent as the guided-prompt direction, yet standardprompt inference blocks abstention-inducing steering $\hat { ( } + \alpha \colon \mathrm { ~ 0 \to 1 \% ) ~ }$ while permitting abstention-suppressing steering $( - \alpha \colon 1 0 {  } 4 \% )$ . The bottleneck is a one-way constraint between the epistemic representation (which exists under both prompts) and the autoregressive output: the model can be pushed toward answering but not toward withholding under standard prompting.

Scope of the causal claim. "Causal" is meant in the interventionist sense: we set hidden states directly $( \mathbf { h }  \mathbf { h } + \alpha \mathbf { v } )$ while holding the input, prompt, and decoding (greedy) fixed, and compare against the α=0 counterfactual on the same pairs, so the effect of the intervention on abstention is identified exactly on the tested distribution. Three standard caveats bound what this licenses. Construct: the intervention is defined on a direction, not on a concept; that v captures epistemic content rather than, e.g., a generic hedging-style axis rests on converging evidence (cross-dataset and cross-modality transfer, the ${ \check { \mathrm { C f } } } {  } \mathrm { C f }$ restriction, and the ill-posed conditions, whose videos are identical to controls and thus exclude visual confounds from those directions). Manifold: large α may push activations off the natural data manifold; the monotone dose-response from small α and, especially, the suppression of naturally occurring abstention under —α indicate the direction modulates the pathway natural behavior uses. Mediation: our design establishes causal sufficiency (inducing and suppressing abstention); a full mediation analysis of natural abstention $( \mathrm { e . g . }$ , activation pātching between matched control/void pairs, for which TRAPSBench's minimal-pair structure is well suited) is left to future work.

Occlusion encodes a domain-general epistemic signal; chaos does not. The occlusion and chaotic void directions are nearly orthogonal $( \overline { { \cos } } = - 0 . 0 5 ;$ Table 24), yet both are causally effective within their own domain (ch→ch: 70%, oc→oc: 46%). The asymmetry in cross-domain transfer therefore reflects direction generality: the occlusion direction captures a transferable “evidence is insufficient" signal that surpasses even the chaotic domain's own direction on chaotic targets (90% vs. 70%), while the chaotic direction encodes a domain-specific feature that is orthogonal to what other domains require.

Table 25: Cross-architecture behavioral results (equal-weight mean over the four datasets; per-dataset detail with ±std in Table 26).
<table><tr><td>Model</td><td>Regime</td><td>Acc %</td><td>AbsRec %</td><td>FalseAbs %</td><td>PECS</td></tr><tr><td>Gemma 4 E4B</td><td>Standard</td><td>47.0</td><td>42.7</td><td>2.8</td><td>.160</td></tr><tr><td></td><td>JSON</td><td>47.3</td><td>46.2</td><td>3.7</td><td>.178</td></tr><tr><td></td><td>Guided</td><td>45.9</td><td>76.1</td><td>25.6</td><td>.215</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>Standard</td><td>35.1</td><td>35.5</td><td>9.6</td><td>.079</td></tr><tr><td></td><td>JSON</td><td>29.8</td><td>14.6</td><td>5.0</td><td>.025</td></tr><tr><td></td><td>Guided</td><td>33.4</td><td>46.1</td><td>12.7</td><td>.101</td></tr></table>

## J Cross-Architecture Replication

The mechanistic analyses in Sections 5.4–5.5 were developed on Qwen3-VL-8B. To test whether the representation-output gap is a model-specific artifact, we replicate the full pipeline (behavioral evaluation via PECS, linear probing, and activation steering) on two additional open-weight families, selected for (i) open weights (required for hidden-state access), (ii) comparable scale, and (iii) organizational distinctness from Qwen3-VL-8B (Alibaba), sharing no pre- or post-training pipeline with it or with each other: Gemma 4 E4B (Google) and LLaVA-NeXT-Video-7B (Vicuna-7B backbone with a CLIP-336 vision encoder). Both models run the identical protocol: three independent end-to-end evaluation runs over all four datasets and three prompt regimes, judged by the same 3-model panel; probing and steering follow the extraction and steering procedures of Appendices H and I.

## J.1 Behavioral Replication: Low PECS

Both families fail to express spontaneous epistemic restraint, and guided prompting unlocks latent abstention with the same signature as the main leaderboard (Table 25). Gemma's AbsRec rises 1.8× from standard to guided (42.7%→76.1%) and LLaVA's 1.3× (35.5%→46.1%), yet guided PECS remains far from saturation for both (0.215 and 0.101).

Per-split behavior mirrors the main findings. Gemma's weakest split is occlusion (guided PEC .124 ± .014; standard .046 ± .013), where the model confidently answers questions whose visual evidence is hidden behind a wall, while its strongest are chaotic (.273 ± .009) and occlusion ill-posed (.243 ± .011). LLaVA shows the occlusion asymmetry as well (guided PECS: occlusion .059 ± .022 vs. occlusion ill-posed .150 ± .006), struggling most on chaotic scenes (23.8% accuracy).

Table 26 gives the full per-dataset breakdown; the chaotic ill-posed rows are from the verification run on the original question set (Appendix P).

## J.2 Probing Replication: The Void/Control Distinction Is Decodable

We repeat the cross-dataset LR-probe transfer protocol of Section 5.4 (best layer per pair, AUROC; Cf→Cf restricts both train and test sets to void samples the model confabulated on, paired with matched controls). Per-pair detail for all three families appears in the main text (Table 3); Table 27 gives the four-path means per transfer type.

The void/control distinction is linearly decodable in both new families and transfers across datasets at levels comparable to Qwen3-VL-8B, including under the Cf→Cf restriction: the models encode the distinction on exactly the inputs where they fail to act on it. Figure 7 makes this dissociation explicit: every Cf→Cf value is computed only on void samples the model confabulated on (0% expressed abstention by construction), yet transfer AUROC sits far above chance in all three families (medians .73–.82).

Table 26: Per-dataset results for the replication families (mean±std over three runs).
<table><tr><td>Dataset</td><td>Regime</td><td>Acc %</td><td>AbsRec %</td><td>FalseAbs %</td><td>PECS</td></tr><tr><td colspan="6">Gemma 4 E4B</td></tr><tr><td>occlusion occlusion occlusion</td><td>standard json guided</td><td> $6 2 . 1 \pm 1 . 5$   $6 1 . 4 \pm 1 . 4$   $5 4 . 8 \pm 1 . 6$ </td><td> $1 1 . 7 \pm 1 . 3$   $1 2 . 2 \pm 1 . 0$   $5 8 . 4 \pm 2 . 2$ </td><td> $4 . 3 \pm 1 . 0$   $5 . 8 \pm 0 . 8$   $3 5 . 8 \pm 1 . 5$ </td><td>.046 ± .013 .039 ± .004</td></tr><tr><td>occl. ill-posed occl. ill-posed occl. ill-posed chaotic</td><td>standard json guided standard</td><td> $6 1 . 4 \pm 1 . 9$   $6 2 . 1 \pm 0 . 6$   $5 6 . 1 \pm 0 . 6$   $3 2 . 4 \pm 2 . 2$ </td><td> $3 9 . 4 \pm 0 . 6$   $5 2 . 5 \pm 1 . 4$   $7 9 . 9 \pm 1 . 2$   $5 6 . 3 \pm 0 . 9$ </td><td> $4 . 7 \pm 0 . 2$   $5 . 4 \pm 1 . 0$   $3 6 . 5 \pm 2 . 7$   $1 . 0 \pm 0 . 5$ </td><td>.124 ± .014 .214 ± .009 .292 ± .014  $. 2 4 3 \pm . 0 1 1$   $. 1 7 9 \pm . 0 1 5$ </td></tr><tr><td>chaotic</td><td>json</td><td> $3 3 . 3 \pm 1 . 1$ </td><td> $5 2 . 6 \pm 0 . 9$ </td><td> $1 . 9 \pm 0 . 1$ </td><td> $. 1 6 9 \pm . 0 0 4$ </td></tr><tr><td>chaotic</td><td>guided</td><td> $3 6 . 3 \pm 1 . 0$ </td><td> $8 9 . 0 \pm 0 . 2 $ </td><td> $1 3 . 8 \pm 1 . 9$ </td><td> $. 2 7 3 \pm . 0 0 9$ </td></tr><tr><td>ch. ill-posed</td><td>standard</td><td> $3 2 . 2 \pm 0 . 8$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td> $6 3 . 5 \pm 0 . 5$ </td><td> $1 . 0 \pm 0 . 6$ </td><td> $. 2 0 1 \pm . 0 0 4$ </td></tr><tr><td>ch. ill-posed</td><td>json</td><td> $3 2 . 5 \pm 1 . 4$ </td><td> $6 7 . 4 \pm 0 . 5$ </td><td> $1 . 7 \pm 0 . 4$ </td><td> $. 2 1 3 \pm . 0 0 8$ </td></tr><tr><td>ch. ill-posed</td><td>guided</td><td> $3 6 . 4 \pm 0 . 7$ </td><td> $7 6 . 9 \pm 1 . 1$ </td><td> $1 6 . 4 \pm 0 . 5$ </td><td> $. 2 2 0 \pm . 0 0 8$ </td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>occlusion</td><td>standard</td><td> $4 8 . 8 \pm 1 . 3$ </td><td> $2 1 . 8 \pm 3 . 6$ </td><td></td><td></td></tr><tr><td>occlusion</td><td></td><td> $3 6 . 8 \pm 1 . 2$ </td><td></td><td> $1 5 . 9 \pm 3 . 1$ </td><td> $. 0 2 9 \pm . 0 2 5$ </td></tr><tr><td></td><td>json</td><td></td><td> $9 . 1 \pm 0 . 2$ </td><td> $5 . 6 \pm 0 . 9$ </td><td> $. 0 1 3 \pm . 0 0 4$ </td></tr><tr><td>occlusion</td><td>guided</td><td> $4 2 . 7 \pm 1 . 7$ </td><td> $3 3 . 7 \pm 1 . 4$ </td><td> $2 0 . 0 \pm 3 . 3$ </td><td> $. 0 5 9 \pm . 0 2 2$ </td></tr><tr><td>occl. ill-posed</td><td>standard</td><td> $4 5 . 4 \pm 0 . 9$ </td><td> $4 2 . 6 \pm 1 . 5$ </td><td> $1 5 . 3 \pm 0 . 4$ </td><td> $. 1 2 4 \pm . 0 0 3$ </td></tr><tr><td>occl. ill-posed</td><td>json</td><td> $3 3 . 3 \pm { 3 . 1 }$ </td><td> $1 1 . 9 \pm 0 . 8$ </td><td> $7 . 9 \pm 0 . 7$ </td><td> $. 0 1 3 \pm . 0 0 5$ </td></tr><tr><td>occl. ill-posed</td><td>guided</td><td> $4 6 . 2 \pm 2 . 5$ </td><td> $4 9 . 7 \pm 1 . 8$ </td><td> $1 7 . 0 \pm 1 . 9$ </td><td> $. 1 5 0 \pm . 0 0 6$ </td></tr><tr><td>chaotic</td><td>standard</td><td> $2 2 . 8 \pm 0 . 5$ </td><td> $2 0 . 5 \pm 1 . 3$ </td><td> $1 . 1 \pm 0 . 3$ </td><td> $. 0 4 4 \pm . 0 0 1$ </td></tr><tr><td>chaotic</td><td>json</td><td> $2 4 . 4 \pm 2 . 0$ </td><td> $9 . 8 \pm 0 . 9$ </td><td> $2 . 1 \pm 0 . 1$ </td><td> $. 0 1 0 \pm . 0 0 2$ </td></tr><tr><td>chaotic</td><td>guided</td><td> $2 3 . 8 \pm 0 . 8$ </td><td> $4 0 . 3 \pm 0 . 7$ </td><td> $2 . 0 \pm 0 . 5$ </td><td> $. 0 9 1 \pm . 0 0 4$ </td></tr><tr><td>ch. ill-posed</td><td>standard</td><td> $2 3 . 5 \pm 2 . 3$ </td><td> $5 6 . 9 \pm 0 . 5$ </td><td> $5 . 9 \pm 0 . 3$ </td><td> $. 1 2 0 \pm . 0 1 1$ </td></tr><tr><td>ch. ill-posed</td><td>json</td><td> $2 4 . 5 \pm 1 . 7$ </td><td> $2 7 . 4 \pm 1 . 2$ </td><td> $4 . 5 \pm 0 . 3$ </td><td> $. 0 5 6 \pm . 0 0 7$ </td></tr><tr><td>ch. ill-posed</td><td>guided</td><td> $2 1 . 0 \pm 1 . 2$ </td><td> $6 0 . 7 \pm 0 . 8$ </td><td> $1 1 . 8 \pm 1 . 5$ </td><td> $. 1 0 3 \pm . 0 0 8$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 27: Four-path probe-transfer means per transfer type (AUC / Cf→Cf).
<table><tr><td></td><td>Qwen3-VL-8B</td><td>Gemma 4 E4B</td><td>LLaVA-Video-7B</td></tr><tr><td>Cross-modality</td><td> $. 8 5 1 ~ / ~ . 8 2 3$ </td><td>.881 / .822</td><td>.913 / .908</td></tr><tr><td>Cross-domain</td><td>.726 / .718</td><td>.861  /  .787</td><td>.758 / .732</td></tr><tr><td>Cross-both</td><td>.795 / .783</td><td>.833 / .725</td><td>.722 / .716</td></tr></table>

![](images/e401f394105bd27f18145afb556ed3334855be4ed0df5542ff7642d168711f86.jpg)  
Figure 7: "Knows but won't $\mathrm { s a y } , \prime \prime$ per transfer pair: Cf→Cf AUROC for the twelve crossdataset transfers of Table 3, computed exclusively on void samples where the model expressed 0% abstention. A purely behavioral account predicts chance (dashed); instead the internal distinction remains decodable in all three families (horizontal bars: medians). Only two pairs approach chance: Qwen's ch\_ip→oc\_ip (.507) and LLaVA's ch→oc\_ip (.482); all others exceed .57.

A caveat on within-split probes. Within-dataset (non-transfer) probes reach AUROC 1.0 from layer 1 in all three families; label-shuffled controls score ≈0.5 (Gemma 0.508, LLaVA 0.516), ruling out leakage. Because control and void videos differ visually, early-layer within-split separation reflects visual content separability rather than epistemic encoding; we therefore rest all claims on cross-dataset transfer and the Cf→Cf restriction, which require a signal that generalizes across taxonomies and distortions.

## J.3 Steering Replication: Causal Control of Abstention

We replicate the activation steering protocol at each family's best layer (Gemma L35; LLaVA L28; Qwen L20), sweeping $\alpha \in \{ \breve { 0 } , \dot { 2 } , 5 , 1 0 \}$ with the same ±α convention (control receives +α to induce abstention; void receives —α to suppress it). The full direction-regime × inference-regime matrix and the direction-family asymmetry for all three families appear in the main text (Table 4). The Gemma and LLaVA prompt-gating rows aggregate the identical eight-path set as Qwen's: the four occlusion-family and four chaotic-family within-domain (direction, target) combinations.

## Three observations:

Single-layer steering causally controls abstention in all three families. Under guided/guided, +α raises control abstention (Gemma 25→57%; LLaVA 16→50%; Qwen 11→60%) and −α lowers void abstention (78→65%; 52→28%; 67→27%).

The direction-family asymmetry replicates in Gemma. Occlusion-family directions dominate chaotic-family directions for abstention induction in Gemma (35→84 vs. 15→29), as in Qwen (11→75 vs. 11→15); in LLaVA the occlusion direction reaches a higher endpoint (19→54 vs. 13→47) but the uplift is comparable, so the asymmetry is weak there. This is consistent with occlusion encoding a more domain-general “evidence is missing" signal in Qwen and Gemma.

The prompt gate is family-specific. The two standard-inference rows isolate the gate. Qwen is fully gated (+α induction ≤1% under standard inference); Gemma is partially gated (17–20% under standard vs. 39–57% under guided); LLaVA is not gated (+α reaches ≥50% under both regimes). Abstention suppression (—α) operates under standard inference in all three families. We therefore scope the unidirectional-gate finding (Section 5.5, Finding 1) to Qwen3-VL, with Gemma exhibiting a partial gate.

For Gemma we additionally swept steering depth (control condition, average over the four splits): layer 1 yields 23→38% abstention at α=10, layer 17 is inert (24→23%), and layer 35 is strongest (24→57%; per-split at α=10: chaotic 23%, chaotic ill-posed 37%, occlusion 70%, occlusion ill-posed 97%). As in Qwen, late-layer intervention, closest to the output projection, is the most effective site, consistent with an output-stage bottleneck acting on a representation formed earlier in the network.

Void-direction geometry is family-specific. Figure 8 shows the void-direction cosine matrices for the replication families at their steering layers (standard-prompt extraction, from the verification re-run; compare Qwen in Figure 5b). The geometry underlying the shared behavioral asymmetry is not conserved. On Gemma, the strongly aligned directions are the textual/ill-posed ones (oc\_ip↔ch\_ip = .52) and within-modality pairs (oc↔oc ip = .48, ch↔ch\_ip = .55), while the visual cross-domain pair is near-orthogonal (oc↔ch = .18); on LLaVĀ all off-diagonals are weak (≤ .32, one negative), including within-domain occlusion (.09). The epistemic-transparency interpretation of Finding 2 (Section 5.5) is thus supported geometrically in Qwen, only partially in Gemma (where domain generality attaches to the ill-posed directions), and largely absent in LLaVA, consistent with LLaVA's weaker direction-family asymmetry in Table 4. We therefore treat the geometry, unlike the behavioral asymmetry, as model-dependent.

For Gemma we also recomputed the full within-domain dose-response at L35 in an independent verification re-run (Table 28; reduced generation budget and N=50 per path, so absolute rates sit below the original run of Table 4, whose endpoints remain authoritative). The qualitative pattern mirrors Qwen (Figure 9): occlusion directions act as an effective restraint knob (control abstention oc→occlusion 24→60, oc→occlusion ill-posed 22→60 at α=0→10) while chaotic directions are flat (≤ 8). In the same re-run, LLaVA's dose-response moves far less in every condition, consistent with its weaker direction-family asymmetry (Table 4) and its unstructured direction geometry (Figure 8).

![](images/a56652324429eee7ea3fc160b39ea4d52776b46bb0bc37fd95c40e310721d832.jpg)  
Figure 8: Void-direction cosine similarity for all three families at their steering layers, on a shared color scale (±0.6; standard-prompt hidden states; Gemma/LLaVA from the verification re-run, Qwen reproduced from Figure 5b for comparison). Gemma's alignments are stronger than Qwen's but attach to different direction pairs; LLaVA shows little structure anywhere.

![](images/c8a91409c20e4b2334c8b18415281b778805a92863d3646e8e806f662b9a1c03.jpg)  
Figure 9: Gemma 4 E4B within-domain dose-response at L35 (verification re-run; data of Table 28), in the format of Figure 5a. Occlusion-family directions (blue) increase overall on control samples while chaotic-family directions (red) stay flat; on void samples, only oc\_ip→occlusion shows strong —α suppression (70→36).

Table 28: Gemma 4 E4B within-domain dose-response at L35 (abstention $\% ;$ standardprompt directions, guided inference, N=50/cell; independent verification re-run with a reduced generation budget; absolute rates sit below the original run). Control receives +α void -α.
<table><tr><td></td><td colspan="4">Control (+α)</td><td colspan="4">Void (−α)</td></tr><tr><td>Direction → target</td><td>0</td><td>2</td><td>5</td><td>10</td><td>0</td><td>2</td><td>5</td><td>10</td></tr><tr><td>ch → chaotic</td><td>2</td><td>2</td><td>6</td><td>4</td><td>70</td><td>62</td><td>72</td><td>60</td></tr><tr><td>ch → chaotic ip</td><td>2</td><td>2</td><td>8</td><td>4</td><td>82</td><td>82</td><td>82</td><td>82</td></tr><tr><td>ch_ip → chaotic</td><td>2</td><td>2</td><td>2</td><td>8</td><td>70</td><td>70</td><td>68</td><td>64</td></tr><tr><td>ch_ip → chaotic ip</td><td>2</td><td>2</td><td>2</td><td>8</td><td>82</td><td>82</td><td>82</td><td>82</td></tr><tr><td>oc → occlusion</td><td>24</td><td>28</td><td>28</td><td>60</td><td>70</td><td>68</td><td>60</td><td>68</td></tr><tr><td>oc → occlusion ip</td><td>22</td><td>28</td><td>28</td><td>60</td><td>88</td><td>88</td><td>86</td><td>86</td></tr><tr><td>oc_ip → occlusion</td><td>24</td><td>24</td><td>34</td><td>32</td><td>70</td><td>64</td><td>62</td><td>36</td></tr><tr><td>oc_ip → occlusion ip</td><td>24</td><td>26</td><td>34</td><td>32</td><td>88</td><td>88</td><td>90</td><td>92</td></tr></table>

## J.4 Summary

All three core claims replicate across both families: (1) VLMs fail to express epistemic restraint (low PECS); (2) VLMs encode the void/control distinction, which transfers across datasets and survives restriction to confabulated samples; (3) a single-layer void direction causally steers abstention; occlusion-family directions transfer most broadly in Qwen and

Table 29: Chaotic sub-scenario breakdown (Standard regime, averaged across all models). Acc = control accuracy, AR = void abstention recall, FA = false abstention on control.
<table><tr><td>Sub-scenario</td><td></td><td>N Answer type</td><td>Acc</td><td>FA AR</td><td>PECS</td><td> $\mathrm { A R } _ { \mathrm { i l l } }$ </td></tr><tr><td>Pachinko</td><td></td><td>50 count (0–3)</td><td>65.6</td><td>0.0 1.3</td><td>0.009</td><td>56.4</td></tr><tr><td>Seesaw</td><td></td><td>200 3-way categorical</td><td>82.9</td><td>0.1 13.1</td><td>0.106</td><td>41.2</td></tr><tr><td>Tumbling Dice 200 6-way color</td><td></td><td></td><td>25.5</td><td>1.8 6.1</td><td>0.011</td><td>55.3</td></tr></table>

Gemma, more weakly in LLaVA. The representation-output gap is therefore not an artifact of one architecture or training pipeline. Two results are model-dependent and scoped accordingly: the unidirectional prompt gate (per-family, main text) and the direction geometry (above).

## K Sub-Scenario Analysis

The aggregate results in the main text average over diverse sub-scenarios within each dataset. Here we disaggregate by sub-scenario to reveal how task structure shapes both accuracy and epistemic behavior. All metrics are averaged across all evaluated models unless otherwise noted. This analysis was computed on the original benchmark manifest (Appendix P).

## K.1 Chaotic Sub-Scenarios

The chaotic dataset comprises three sub-scenarios with fundamentally different task structures (Table 29).

Seesaw Sorter achieves the highest accuracy (82.9%) and the only meaningful spontaneous abstention rate (13.1% AR). Decomposing by ground-truth outcome reveals the source: models achieve \~100% accuracy on “left" and "right" outcomes but only \~62% on “elsewhere." The two bins create a strong visual affordance; when the ball lands outside both, models can recognize the outcome violates their expected schema, triggering uncertainty. Seesaw is the best-calibrated sub-scenario.

Pachinko Waterfall has medium accuracy (65.6%) but near-zero abstention (1.3% AR), yielding the lowest PECS. Ball counting is a concrete, well-practiced task, and the small discrete answer space (0–3) makes guessing appear reasonable. Even guided prompting barely helps (AR rises to only 6.6%).

Tumbling Dice has the lowest accuracy (25.5%, barely above 16.7% chance for 6-way classification) yet also low abstention (6.1% AR), the worst calibration of any sub-scenario. Models are confidently wrong. Color identification from small die faces in video is genuinely difficult; accuracy varies by color (red: 34%, green: 38% vs. cyan: 16%, orange: 22%), likely reflecting differences in perceptual salience. Despite being the hardest task, models rarely acknowledge uncertainty because color identification feels like a factual rather than uncertain question.

Guided prompting uplift. The guided regime amplifies abstention differentially across sub-scenarios (Table 30). Tumbling dice shows the largest AR uplift (+18.3pp) but also the largest false abstention increase (+4.2pp); the prompt helps but causes over-abstention on answerable questions. Seesaw responds cleanly (+13.9pp AR with only +1.0pp FA). Pachinko barely responds (+5.3pp), suggesting that counting tasks are particularly resistant to epistemic prompting.

Visual vs. textual unanswerability gap. The gap between visual (chaotic baseline) and textual (ill-posed) abstention rates varies strikingly by sub-scenario. Pachinko shows the largest ratio $( 1 . 3 \%  5 6 . 4 \% , a 4 4 \times \mathrm { g a p } )$ : the visual counting task feels completely tractable even when the truncated video makes it impossible. Seesaw has the smallest gap (3.1 ×)

Table 30: Standard → Guided prompting uplift by chaotic sub-scenario (averaged across models).
<table><tr><td rowspan="2">Sub-scenario</td><td colspan="2">Standard</td><td colspan="2">Guided</td></tr><tr><td>AR</td><td>FA</td><td>AR</td><td>FA</td></tr><tr><td>Pachinko</td><td>1.3</td><td>0.0</td><td>6.6</td><td>0.0</td></tr><tr><td>Seesaw</td><td>13.1</td><td>0.1</td><td>27.0</td><td>1.1</td></tr><tr><td>Tumbling Dice</td><td>6.1</td><td>1.8</td><td>24.4</td><td>6.0</td></tr></table>

Table 31: Occlusion sub-scenario breakdown (Standard regime, averaged across all models). Sub-scenarios sorted by accuracy. "Hidden" sub-scenarios contain an explicit visual occluder.
<table><tr><td>Sub-scenario</td><td>Answer type</td><td>Acc</td><td>FA</td><td>AR</td><td>G-AR</td></tr><tr><td>Deflect</td><td>direction (L/R)</td><td>24.3</td><td>0.0</td><td>4.3</td><td>31.0</td></tr><tr><td>Hidden collision</td><td>binary (Y/N)</td><td>36.2</td><td>0.0</td><td>3.8</td><td>49.0</td></tr><tr><td>Hidden drop</td><td>side (L/R)</td><td>44.3</td><td>4.3</td><td>35.7</td><td>73.3</td></tr><tr><td>Stack</td><td>count range</td><td>45.7</td><td>0.5</td><td>1.4</td><td>29.0</td></tr><tr><td>Bounce</td><td>binary (Y/N)</td><td>51.0</td><td>1.9</td><td>7.6</td><td>51.0</td></tr><tr><td>Spring</td><td>binary (Y/N)</td><td>56.2</td><td>0.0</td><td>0.0</td><td>12.9</td></tr><tr><td>Catch</td><td>binary (Y/N)</td><td>61.4</td><td>0.0</td><td>0.5</td><td>53.8</td></tr><tr><td>Pendulum</td><td>binary (Y/N)</td><td>61.9</td><td>0.0</td><td>3.3</td><td>48.6</td></tr><tr><td>Chain</td><td>binary (Y/N)</td><td>68.1</td><td>0.0</td><td>11.9</td><td>54.8</td></tr><tr><td>Balance</td><td>side (L/R/bal.)</td><td>73.8</td><td>10.5</td><td>10.5</td><td>37.6</td></tr><tr><td>Topple</td><td>binary (Y/N)</td><td>74.3</td><td>0.0</td><td>1.4</td><td>4.8</td></tr><tr><td>Hidden race</td><td>which-first (color)</td><td>75.2</td><td>9.0</td><td>41.4</td><td>81.4</td></tr><tr><td>Slide</td><td>binary (Y/N)</td><td>78.1</td><td>0.0</td><td>0.0</td><td>3.8</td></tr><tr><td>Launch</td><td>binary (Y/N)</td><td>81.0</td><td>0.0</td><td>13.3</td><td>69.5</td></tr><tr><td>Tower</td><td>binary (Y/N)</td><td>91.4</td><td>0.5</td><td>4.3</td><td>21.9</td></tr></table>

Contains an explicit visual occluder in the void condition. G-AR = Guided regime AR.

because its visual uncertainty signal (ball landing outside bins) partially resembles the textual signal (question about a non-existent element).

## K.2 Occlusion Sub-Scenarios

The occlusion dataset comprises 15 sub-scenarios spanning diverse physics phenomena. Accuracy ranges from 24% (deflect) to 91% (tower), while spontaneous abstention ranges from 0% (slide, spring) to 41% (hidden\_race) (Table 31).

Hidden vs. observable scenarios. The three sub-scenarios with an explicit visual occluder (hidden\_collision, hidden\_drop, hidden\_race) show markedly different abstention behavior: 27.0% AR in Standard vs. 4.9% for the 12 observable scenarios. With guided prompting, this gap widens further (67.9% vs. 34.9%). This suggests that models can partially recognize when a visible occlusion event interrupts the information they need, but struggle to detect subtler information gaps where the rēlevant physics simply cannot be inferred from the visible evidence.

Extreme cases. At one extreme, hidden\_race (two balls entering a tunnel, one exits first) achieves 41.4% spontaneous AR, as models frequently recognize they cannot determine the winner without seeing inside the tunnel. At the other extreme, slide and spring show 0% AR: these binary questions ("Does the block slide off?", "Does the ball clear the barrier?") elicit confident answers from every model. Guided prompting has almost no effect on these scenarios (slide: 3.8%, topple: 4.8%), suggesting that binary physics questions with strong priors are particularly resistant to epistemic prompting.

Guided prompting uplift. The response to guided prompting varies by more than an order of magnitude across sub-scenarios: launch gains +56.2pp AR (the explicit framing “if unanswerable" enables models to acknowledge the interrupted trajectory), while slide and topple gain only +3–5pp. Scenarios where the model can recognize but not resolve an information gap respond most to prompting; scenarios where the model holds a strong physics prior ('blocks don't slide unless pushed hard") resist it.

![](images/a1a272885aa1ad0518215675f5c2926e52f5483ba49320d95efe7eb6a5c0ffec.jpg)  
Figure 10: Occlusion sub-scenario abstention recall (%) for 9 video-native models under Standard (left) and Guided (right) prompting. Sub-scenarios are sorted by average Standard AbsRec (hardest at left). Slide, Špring, and Stack remain near-zero even with guided prompting, while Race and Drop respond strongly.

## K.3 Mechanistic Hypotheses

The sub-scenario analysis suggests three factors that determine whether a model will spontaneously abstain:

1. Schema violation detection. Models abstain most readily when the observed outcome violates their expected schema. Seesaw's “elsewhere" outcome (ball lands outside both bins) and hidden\_race's tunnel (winner is invisible) create recognizable expectation violations. In contrast, tumbling dice and pachinko produce outcomes thāt look normal (a color, a count) even when the model cannot actually determine them.

2. Answer-space size and task framing. Small answer spaces (binary Y/N, small counts) suppress abstention because guessing carries little apparent penalty. Models treat “Does X happen?" as a factual question demanding an answer, not as a question about whether they have sufficient evidence. The 6-way tumbling dice task should in principle produce more uncertainty than binary tasks, but color identification is framed as perceptual fact rather than inference under uncertainty.

3. Occluder salience. Explicit visual occluders (tunnels, walls blocking view) trigger abstention far more efectively than implicit information gaps (truncated video, chaotic sensitivity). This parallels the visual-textual gap: models detect linguistic markers of unanswerability ("What material are the triangles made of?" when no triangles exist) much more readily than visual markers (video that lacks the information needed to answer). The hierarchy is: textual impossibility > visible occlusion > implicit visual insufficiency.

Figure 10 provides a per-model view of occlusion sub-scenario difficulty under both Standard and Ġuided regimes, revealing which models and scenarios benefit most from explicit epistemic prompting.

## L Additional Qualitative Examples

## L.1 False Abstention on Control (Inverted Calibration)

## Gemini 2.5 Pro, Chaotic (Tumbling Dice), Guided regime.

Question: "Which color face landed up on the first die?"

Control response (answerable; model falsely abstains):

"At 00:00.400, the first die appears with its green face on top. [.. .] Between 00:00.800 and 00:01.000, the first die lands on the step and is immediately struck and covered by a second die. [...] Because the die's top face is never visible after it comes to its final rest, and its orientation changed during the collision, its final upward-facing color cannot be determined. I don't know.

Void response (unanswerable; model paradoxically gives a confident answer):

"The first die, which is the lowest in the falling stack, initially has a green face on top. It impacts the third step from the bottom. Upon impact, it rotates forward [.. .] The die comes to rest with the blue face pointing up. Blue."

This example shows inverted calibration: the model is cautious on the answerable question but confidênt on the unanswerable one. The guided prompt produces overcautious behavior on control while failing to prevent hallucination on void.

## L.2 Hallucinated Physical Trajectory

## Gemini 2.5 Pro, Occlusion (Object Motion), Standard regime.

Void response (object motion is fully occluded):

"An object enters the frame from the right at 00:01.000 and travels horizontally to the left. At 00:03.200, the object makes contact with the wall on the left side of the scene. Following the collision, the object rebounds and moves back towards the right. Yes, the object reaches the wall."

The model fabricates an entire trajectory with specific timestamps, direction, collision, and rebound, none of which is visible in the video.

## L.3 Visual vs. Textual Unanswerability: Seesaw Example

Qwen3-VL Instruct, Chaotic (Seesaw Sorter), Standard regime.

Visual void question: "Where did the ball end up – left bin (red), right bin (blue), or somewhere else?" The chaotic physics makes the trajectory unpredictable. Response (fails to abstain):

"The yellow ball lands on the wooden plank slightly to the right [.. . ] it will cause the plank to tilt clockwise [.. . ] toward the right bin.

Ill-posed question (same video): "What is the weight of the red color?"Response (correctly abstains):

"The question 'What is the weight of the red color?' is physically invalid. Color is a visual property resulting from light wavelengths and does not have mass or weight."

The contrast reveals two distinct reasoning pathways. Textual unanswerability triggers language-level semantic checking: “weight of a color" is a category error detectable through pattern matching against common-sense knowledge. Visual unanswerability requires the model to recognize that the visual evidence itself is insufficient, that chaotic dynamics make the trajectory genuinely unpredictable. Models lack this metacognitive capacity, instead confabulating plausible-sounding answers with fabricated physical details.

![](images/87ad69548eab54bdb1c22621d1a9715ecff196d468cbfd831251d9440ddcfae7.jpg)  
Figure 11: Per-model abstention recall, Standard (hollow) vs. Guided (filled): the per-model detail behind the rightward trajectories of Figure 3a.

## L.4 Architecture-Dependent Reasoning Effects

Gemini 2.5 Flash vs. Flash NT (Seesaw Sorter, void video): Flash NT fabricates: "The ball rolls onto the plank from the right [.. . ] The ball continues to roll rightward and falls into the blue bin." Flash (thinking) correctly recognizes: "The ball is not visible in the provided frames. Therefore, its trajectory cañnot be determined." The reasoning process helps Gemini verify its observations before committing.

Qwen3-VL Think vs. Instruct (Seesaw Sorter, void video): Think fabricates: "the purple block is positioned slightly left of the seesaw's center [.. . ] The ball will roll down the tilted seesaw toward the left side."Instruct correctly hesitates: “there is no visible force or slope pushing it left or right [.. . ] it will not enter either bin." For Qwen3-VL, reasoning amplifies confabulation: the thinking trace constructs a plausible physical narrative and then commits to it, whereas the non-thinking variant stays closer to what is actually visible.

## L.5 Prompt Regime Flipping Behavior

Seesaw Sorter, Gemini 3.1 Flash Lite. On the void video under Standard regime, the model fabricates: "the seesaw is slightly tilted toward the right (blue bin) [.. .] Final Answer: Right (blue bin)." Under Guided regime on the same void video: "there is no net torque to cause rotation toward either side [... ] I don't know." Both regimes produce correct answers on control. The guided prompt flips void behavior without degrading control performance.

## M PECS Metric Robustness

A well-designed calibration metric should be robust against degenerate strategies: no trivial policy should achieve a high score. We verify this for PECS both analytically and empirically.

Figure 3a (main text) visualizes the metric over all sixteen models: because PECS is a product, its level sets are hyperbolas in (Acc, J) space, and a high score requires jointly moving up and right, into the region no current model occupies. In that figure, hollow → filled markers trace Standard → Guided prompting and rings mark each family's best guided model. Models cluster in the capable-but-indiscriminate upper-left. The guided-regime Pareto frontier reduces to two models (Gemini 3.1 Pro and its R-Low variant, which together dominate all others), so across models capability and discrimination are not in tension;

Table 32: Degenerate strategies and their PECS scores. All yield PECS = 0 through at least one of two safeguards: $\mathrm { A c c } = 0$ or $J \leq 0 .$
<table><tr><td>Strategy</td><td>Acc</td><td>FA</td><td>AR</td><td>J</td><td>Why PECS= 0</td></tr><tr><td>Always abstain</td><td>0%</td><td>100%</td><td>100%</td><td>0</td><td> $\operatorname { A c c } = 0$ </td></tr><tr><td>Never abstain</td><td>any</td><td>0%</td><td>0%</td><td>0</td><td> $J = 0$ </td></tr><tr><td>Random (50%)</td><td>any</td><td>50%</td><td>50%</td><td>0</td><td> $J = 0$ </td></tr><tr><td>Perfect answerer</td><td>100%</td><td>0%</td><td>0%</td><td>0</td><td> $J = 0$ </td></tr><tr><td>Inverse oracle</td><td>0%</td><td>100%</td><td>0%</td><td>&lt;0</td><td>Both terms zero</td></tr><tr><td>Confidence noise</td><td>any</td><td>≈p</td><td>≈p</td><td>≈0J ≈ 0</td><td></td></tr></table>

Table 33: Comparison of PECS vs. the naive Acc × AbsRec metric (Guided regime). The naive metric inlates scores for models with high false abstention. Showing cases where the discrepancy exceeds 0.05.
<table><tr><td>Model</td><td>Dataset</td><td>Naive</td><td>PECS</td><td>FA</td></tr><tr><td>Qwen3-VL 8B</td><td>Chaotic Ill-Posed</td><td>0.328</td><td>0.208</td><td>29.1%</td></tr><tr><td>Qwen3-VL Think</td><td>Occ. Ill-Posed</td><td>0.581</td><td>0.518</td><td>8.6%</td></tr><tr><td>Qwen3-VL Think</td><td>Occlusion</td><td>0.510</td><td>0.458</td><td>7.1%</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>Occlusion</td><td>0.702</td><td>0.645</td><td>7.4%</td></tr><tr><td>Gemini 3.1 Pro R-Low</td><td>Occ. Ill-Posed</td><td>0.586</td><td>0.532</td><td>6.9%</td></tr><tr><td>Gemini 3.1 Pro</td><td>Occlusion</td><td>0.683</td><td>0.630</td><td>6.6%</td></tr><tr><td>Gemini 2.5 Pro</td><td>Occlusion</td><td>0.639</td><td>0.586</td><td>6.8%</td></tr><tr><td>Gemini 2.5 Pro</td><td>Occ. Ill-Posed</td><td>0.504</td><td>0.453</td><td>6.6%</td></tr><tr><td>Gemini 2.5 Flash</td><td>Occlusion</td><td>0.567</td><td>0.516</td><td>6.9%</td></tr></table>

trade-offs appear only within models, as bent trajectories: Gemma 4 E4B and Qwen3-VL-8B pay accuracy and FalseAbs for their AbsRec gains (Figure 11 shows each model's AbsRec shift). The contours illustrate the metric's geometry at aggregate $( \operatorname { A c c } , J ) ;$ reported PECS (Table 1) is computed per dataset and averaged, so values differ slightly.

## M.1 Analytical Robustness

PECS decomposes into two multiplicative factors: task competence (Acc) and epistemic discrimination $( J = \mathrm { A b s R e c - F a l s e A b s } ,$ clamped at zero). Table 32 shows that six canonical degenerate strategies all yield PECS = 0.

The key insight is that the Youden's J statistic measures discriminability: any strategy that treats answerable and unanswerable items identically yields J = 0, zeroing PÉCS regardless of accuracy. Simultaneously, the Acc term means a model cannot achieve high PECS by simply abstaining on everything. These two safeguards are complementary: the Acc factor prevents abstention gaming, while the J factor prevents answer-everything or random strategies.

## M.2 Comparison with Alternative Metrics

PECS's J-statistic penalty makes a practical difference. Table 33 compares PECS with the naive product Acc × AbsRec (which omits the false-abstention penalty). Qwen3-VL 8B under Guided prompting provides the clearest illustration: on Chaotic Ill-Posed, it achieves $\mathrm { A c c } \times \mathrm { A b } \mathbf { \dot { s } } \mathrm { R e c } \overset { \mathbf { \cdot } } { = } 0 . 3 2 8$ but $\mathrm { P E C S } = 0 . 2 0 8$ , because its 29.1% false abstention rate (induced by the guided prompt) penalizes the J term. Similarly, Gemini 3.1 Pro R-Low on Guided Occlusion achieves 0.702 naive but 0.645 PECS due to 7.4% false abstention. Across the 56 main-family Guided-regime model–dataset combinations, 16 show the naive metric exceeding PECS by more than 0.03, all driven by false abstention that the naive metric ignores; the replication families show the same pattern under guided prompting (Appendix J).

![](images/4f3f7ba6f2721a10318936e4be65bea6be4a24fab7df26fcae3f01cd4e562908.jpg)  
Figure 12: Confabulation taxonomy prevalence by reasoning pair (～100 void failures per model). HP and II rates are near-universal. The Doubt axis (rightmost) is architecturedependent: Gemini Flash converts doubt to abstention; Qwen3-VL Think does not.

## M.3 PECS Zero Boundary

In the actual evaluation data, 1 of the 56 main-family Standard-regime model-dataset combinations achieves exactly PECS = 0 (where AbsRec ≤ FalseAbs): GPT-5 Mini on the chaotic baseline, where abstention recall is near-zero. In the JSON regime, 2 of the 56 combinations hit the zero boundary. The Guided regime, by contrast, produces zero PECS-zero entries; under the explicit abstention prompt, all models achieve at least some epistemic discrimination. This confirms that PECS's clamping at zero effectively identifies models with no epistemic discrimination.

## N Additional Observations

Reasoning on vs. off in PECS space. Figure 4a (main text) plots the reasoning-paired variants of Table 1 as trajectories in (Acc, J) space; the Gemini 3.1 reasoning-budget change is nearly inert, and GPT-5's rising effort bends rightward only at Pro. Models without a reasoning-paired variant (Qwen3-VL-8B, Gemma, LLaVA, GPT-5 Mini) are omitted there. The divergent arrow directions restate Section 5.3 geometrically: whether chain-of-thought helps or harms epistemic restraint tracks how the reasoning was trained, not reasoning per se.

Image-only models and structured output. The GPT-5 models process individual frames rather than native video. On chaotic scenarios, they achieve strong accuracy but near-zero standard-regime AbsRec, confirming that epistemic calibration is partially decoupled from task competence.

JSON as an implicit abstention cue. Requiring structured JSON output acts as a weak but consistent implicit abstention cue: for most models, JSON PECS falls between standard and guided regimes (e.g., Gemini 2.5 Flash on Chaotic: 0.222 → 0.372 → 0.436). We hypothesize that the schema constraint forces the model to commit to a typed field for each answer, making fabrication more salient than in free-form text where hedging language can mask uncertainty. This effect is modest compared to explicit guided prompting, but it is notable because it requires no task-specific instruction, only a generic output format.

## O Confabulation Taxonomy

When models fail to abstain on unanswerable (void) scenarios, what do their responses actually look like? To move beyond qualitative observation (Appendix L), we introduce a multi-label confabulation taxonomy grounded in formal epistemic logic, and apply it systematically across six reasoning-paired models.

## O.1 Taxonomy Definition

We define three independent boolean axes, each targeting a distinct failure mode. Because they are independent, a single response can exhibit any combination.

1. Hallucinated Premise (HP). The model asserts a factual observation that is not supported by the visual evidence. Formally, the model claims premise p when the evidence E does not entail $p \left( E \not \vdash p \right)$ . Example: "The ball bounces off the left wall at 00:02.4" when no ball motion is visible in the video.

2. Invalid Inference (II). The model's conclusion does not follow from its stated premises, even if those premises were true. Formally, the premises $p _ { 1 } , \ldots , p _ { n }$ do not entail the conclusion $\bar { q ^ { \prime } } ( p _ { 1 } , \dots , p _ { n } \not \vdash q )$ . Example: "The seesaw tilts slightly right, therefore the ball ends up in the left bin."

3. Epistemic Surrender (ES). The model explicitly acknowledges that it cannot determine the answer $( \neg K ( q ) )$ but then asserts q anyway, a form of Moore's Paradox (“I don't know, but the answer is $\mathbf { \ } ^ { \mathrm { ~ X ~ } \prime \prime } )$ . Example: "It's impossible to tell from this angle, but based on the trajectory the die shows green."

Additionally, for models with reasoning traces (thinking-enabled variants), we annotate a fourth axis:

4. Doubt in Reasoning (Doubt). The model's internal reasoning trace expresses uncertainty, hedging, or doubt about the visual evidence, but the final output does not reflect this doubt. For models with explicit reasoning traces (Gemini Flash, Gemini 3.1 Pro, Qwen3-VL Think) this axis is assessed on the trace; for the remaining variants it is assessed on the visible output.

## O.2 Classification Method

We sample ～100 void-scenario failures per model from run 1 of the standard-regime evaluation (independent random sampling per model). A failure is defined as a void response where the model did not abstain (i.e., it provided a definitive answer to an unanswerable question). For models with reasoning traces (Gemini Flash, Gemini 3.1 Pro, Qwen3-VL Think), the full trace is provided to the classifier; for non-thinking variants (Flash NT, Qwen Instruct) and Gemini 3.1 Pro R-Low, only the output is classified.

Classification is performed by Claude Opus 4.6 with structured JSON output (guided decoding), using the taxonomy definitions above. Each response is classified independently on all applicable axes.

## O.3 Human Validation of the Judge

Because the taxonomy involves subjective metacognitive assessment, we validate the judge against human annotation. The validated subset is drawn from the judge's standard-regime classification set (n=400: ～100 void failures from each of the four reasoning-paired models Gemini 2.5 Flash, Gemini 2.5 Flash NT, Qwen3-VL Think, and Qwen3-VL Instruct; the replication families are not part of this set). Two of the authors independently re-annotated 100 sampled failures on alí four axes (400 judgments per annotator), blind to the judge's labels.

Overall agreement between the two annotators is 375/400 = 93.8%, setting a human-human ceiling of \~94%. Pooled across all four axes, judge accuracy against the two annotators is 93.1%; performance is at or near the human ceiling on three axes (Table 34; the Invalid-Inference accuracy of 0.945 is the mean over the two annotators, 0.91 and 0.98). The Doubt in Reasoning axis, on which the CoT-override claim (Section 5.3) rests, shows judge F1 = 0.91 against human labels and substantial-to-strong human-human reliability (κ = 0.83).

The one weak axis is Epistemic Surrender: the judge is highly conservative (precision 1.00, recall 0.38; it produced zero false alarms but missed 8 of 13 human-labeled cases, with identical confusion matrices against both annotators). ES is also the lowest-prevalence axis (≤9% of failures) and is not used in any core argument; we therefore treat ES prevalence figures as lower bounds.

Table 34: Judge validation. Left: judge performance against pooled human labels (200 judgments per category). Right: human-human agreement on corrected labels.
<table><tr><td></td><td colspan="4">Judge vs. human</td><td colspan="2">Human ↔ human</td></tr><tr><td>Category</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td><td>% agree</td><td>Cohen&#x27;s κ</td></tr><tr><td>Hallucinated Premise</td><td>0.97</td><td>0.99</td><td>0.98</td><td>0.98</td><td>98.0</td><td>*</td></tr><tr><td>Invalid Inference</td><td>0.945</td><td>0.98</td><td>0.95</td><td>0.96</td><td>93.0</td><td>0.79</td></tr><tr><td>Epistemic Surrender</td><td>0.92</td><td>1.00</td><td>0.38</td><td>0.56</td><td>92.0</td><td>0.65</td></tr><tr><td>Doubt in Reasoning</td><td>0.89</td><td>0.93</td><td>0.89</td><td>0.91</td><td>92.0</td><td>0.83</td></tr></table>

κ is degenerate for HP: 98/100 labels are positive, leaving near-zero label variance; we report raw agreement instead.

Table 35: Confabulation taxonomy prevalence across six reasoning-paired models (～100 void failures per model, standard regime, independent sampling). HP and II are nearuniversal; ES is rare. Doubt is assessed on the reasoning trace for thinking-enabled models (marked with †) and on the visible output otherwise.
<table><tr><td>Model</td><td>n</td><td>HP</td><td>I</td><td>HP∩II</td><td>ES</td><td>Doubt</td></tr><tr><td>Gem 3.1 Pro†</td><td>98</td><td>87%</td><td>83%</td><td>76%</td><td>2%</td><td>17%</td></tr><tr><td>Gem 3.1 Pro R-Low</td><td>100</td><td>96%</td><td>89%</td><td>85%</td><td>1%</td><td>16%</td></tr><tr><td>Gem 2.5 Flash†</td><td>98</td><td>93%</td><td>89%</td><td>84%</td><td>2%</td><td>20%</td></tr><tr><td>Gem 2.5 Flash NT</td><td>100</td><td>99%</td><td>74%</td><td>73%</td><td>0%</td><td>2%</td></tr><tr><td>Qwen3-VL Think†</td><td>100</td><td>87%</td><td>88%</td><td>75%</td><td>1%</td><td>24%</td></tr><tr><td>Qwen3-VL Instruct</td><td>100</td><td>95%</td><td>85%</td><td>81%</td><td>9%</td><td>16%</td></tr></table>

Table 35 reveals several findings (see also Figure 12 in the main text):

Hallucinated premises are near-universal. Across all six models, 87-99% of nonabstaining void responses contain at least one fabricated visual observation (HP). Models do not merely draw wrong conclusions from what they see; they fabricate evidence that does not exist in the video. This is consistent across the evaluated architectures and reasoning modes, indicating that evidence fabrication is a recurring failure mode across the six evaluated models on unanswerable visual inputs.

Invalid inference compounds hallucination. 74–89% of responses also contain invalid inferences (II), and the HP∩II overlap is large (73–85%), meaning most confabulations involve both fabricated premises and faulty reasoning from those premises. The co-occurrence suggests a common generative pattern: the model first hallucinates a plausible physical observation, then constructs an internally consistent (but unsound) narrative around it.

Epistemic surrender is rare. Only 0–9% of responses exhibit Moore's Paradox (ES). When models confabulate, they typically do so with full commitment, rarely hedging or acknowledging doubt in the output (and the judge's conservatism makes these figures lower bounds; Appendix O.3). The exception is Qwen3-VL Instruct (9%), whose non-thinking responses occasionally include hedging language that still terminates in a definitive answer.

Reasoning traces reveal suppressed doubt. Among thinking-enabled models, 17-24% of reasoning traces express internal doubt about the visual evidence (Doubt), yet the final output overrides this doubt to provide a definitive answer. This suppressed-doubt phenomenon parallels the representation-output gap identified via probing (Section 5.4): models encode epistemic uncertainty internally but fail to express it in their outputs.

Crucially, the doubt axis differentiates the reasoning effect across architectures. Gemini Flash (thinking) shows 20% doubt and does convert some of this into improved abstention (AbsRec increases by 4–13pp over Flash NT; Section 5.3). Qwen3-VL Think shows the highest doubt rate (24%) but fails to convert it into abstention; its AbsRec is actually lower than Instruct's on several datasets. This quantitatively confirms that CoT reasoning can either help or hurt epistemic calibration, plausibly depending on whether the training procedure teaches the model to act on its own doubt or to override it.

## P Benchmark Correction and Evaluation Provenance

The headline aggregates in this paper (Table 1 and the per-dataset tables of Appendix F) are computed on the released version of TRAPSBench (202 occlusion and 500 chaotic control/void pairs; Table 6). This appendix documents the data-quality review conducted between acceptance and the camera-ready, the re-evaluation, and which results derive from which evaluation.

What the review found and fixed. A pair-by-pair human review of the occlusion manifest (all 226 original pairs, via a tagging interface) found two classes of defect: (i) duplicated control videos, in which the same rendered clip served multiple nominally distinct scenarios (226 pairs mapped to 147 unique control clips), and (ii) question/answer pairs whose premise or ground truth did not match the rendered clip. The review kept 180 pairs unchanged, corrected 22 (rewriting the question, choices, or control ground truth to match the visible physics), and discardeð 24 whose defects were unfixable, yielding the released 202-pair occlusion set (each pair with its own control and void video file). In the chaotic taxonomy, baseline (control/void) questions were made temporally precise (e.g., asking about the first die to land rather than an ambiguous ordinal) and the 50-pair Plinko scenario family was restored, yielding the released 500-pair chaotic set. The chaotic ill-posed question set has not been updated since the original benchmark; the released benchmark ships the original ill-posed questions, and we therefore reuse the original evaluation results for that split.

Re-evaluation. All sixteen models were re-evaluated on the corrected occlusion, occlusionill-posed, and chaotic splits under the protocol of Section 4 (three prompt regimes, the same 3-model judge panel, three independent runs per API-served model). Qwen3-VL 8B was re-evaluated locally with a single greedy, deterministic pass in bf16 without quantization, using 16 uniformly sampled frames per clip; this differs from the original run's 4-bit NF4, 25-frame configuration. The chaotic ill-posed column of every aggregate reuses the original evaluation (450 of the 500 released ill-posed items; the 50 restored Plinko items have no ill-posed measurements), taken from the original three-run means for the fourteen main-family models (thirteen API-served and Qwen3-VL 8B) and from the verification run for Gemmå and LLaVA. Because the re-evaluation necessarily occurred months after the original runs, aggregate differences from previously reported values reflect the benchmark correction, evolution of API-served model endpoints over the intervening period, and harness alignment jointly; we do not attribute individual differences to any single cause.

Analyses reported entirely on the original evaluation. Ratio-based comparisons require both sides of a contrast to come from a single evaluation epoch, and the released chãotic ill-posed questions carry only original-evaluation measurements. The visual-vs-textual asymmetry analysis is therefore reported entirely on the original evaluation: Table 2, Figure 3b, and Figure 4b, together with the asymmetry ratios and doubt-conversion deltas quoted in Sections 5.2–5.3, the abstract, and the conclusion, use original-evaluation AbsRec values, so their absolute values intentionally differ from the corrected per-dataset tables of Appendix F. The activation-steering analyses (Section 5.5, Appendix I), the confabulation taxonomy (Appendix O), and the sub-scenario analysis (Appendix K) likewise derive from the original evaluation. Evaluation-volume figures in Section 4 describe the nominal per-model protocol (1,404 pairs, three regimes, three runs); the printed aggregates reuse original-evaluation chaotic ill-posed measurements (450 items; a single verification run for

Gemma and LLaVA), so each printed aggregate rests on 1,354 evaluated pairs per model, regime, and run rather than the nominal 1,404.

Findings unaffected. The paper's mechanistic and qualitative conclusions hold on the corrected benchmark: spontaneous restraint remains poor in all five families; guided prompting unlocks latent abstention in every model, with a larger effect than originally measured; the void/control distinction remains linearly decodable and transfers across datasets, including under the confabulation restriction (probe hidden states were re-extracted on the corrected occlusion set; Cf→Cf values for chãotic-test transfers are from the original run); activation steering results are unchanged (steering operates on sampled pairs insensitive to the manifest correction, and its corrected recompute was blocked by a build-environment incompatibility); and the confabulation taxonomy and judge validation are sample-level analyses unaffected by the correction. Between-family orderings are preserved in every table cell we compared.

Findings that changed. Leaderboard placement changed: the best standard-regime PECS is now Gemini 2.5 Flash rather than Qwen3-VL Instruct, and guided-regime PEČS rose for every API-served model, so the guided frontier now clears the 0.4 iso-PECS contour for nine models. Qwen3-VL 8B's guided-regime gain largely disappears under the corrected, greedydecoded configuration (PECS .142 standard vs. .155 guided), placing it below Gemma 4 E4B under guided prompting; the representation-output gap findings for this model (Sections 5.4 and 5.5) are unaffected.