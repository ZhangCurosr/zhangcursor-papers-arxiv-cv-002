# SPARED: Reasoning-Based AI-Generated Image Detection via Adversarially Edited Data

Yicheng Bao<sup>1</sup>, Xiahui Guo<sup>1</sup>, Xuhong Wang<sup>2,\*</sup>, Xin Tan<sup>1,2,\*</sup>

<sup>1</sup>East China Normal University <sup>2</sup>Shanghai Artificial Intelligence Laboratory Co-corresponding authors

## Abstract

Detecting AI-generated images is only half the task: a deployed detector must also justify its verdict, yet existing detectors inherit three failure modes from their training data: real and fake images collected from diferent sources invite provenance shortcuts, supervised explanation corpora teach templated rationales, and a static forgery corpus leaves the decision boundary standing still while generators keep moving. We introduce SPARED, an adversarial reinforcement learning framework that pits two heterogeneous models against each other. A difusion image editor learns to edit real photographs into fake counterparts of those same photographs that fool the current detector, while a reasoning MLLM learns to expose them with a verdict grounded in free-form reasoning. Both rewards are shortcut-proof by design: the attacker is credited only when its edit is faithfully executed, and the defender only when its verdict is correct. As the two models alternate, each round’s attacker regenerates a harder training pool aimed at the current detector’s blind spots, so the detector must generalize rather than memorize any fixed artifact distribution. Although the explanation is never rewarded, its quality rises round over round as a side efect of accuracy-only training. A detector trained within this loop improves monotonically across rounds on each of three external benchmarks.

## Introduction

Generalizable, explainable AI-generated image (AIGI) detection takes a single image (a real camera photograph, a fully AI-generated image, or a locally AI-edited photograph) and outputs a real/fake verdict together with a natural-language account of which regions or cues make the image look synthetic. This judge-style formulation, exemplified by recent MLLM-as-a-judge deepfake benchmarks (Kuckreja et al. 2026) and explainable forensics models (Zhou et al. 2025; Huang et al. 2025c), matters for three reasons. First, moderation must both flag a fabricated image and justify the flag: an unexplained label is easy to dismiss, and a wrong flag on a realistic photo can itself trigger misinformation. Second, deployed detectors meet fakes from an open, growing population of generators (Park and Owens 2025; Huang et al. 2026), so what decides practical value is generalization beyond a fixed forgery distribution, not accuracy on it. Third, public detectors can be targeted by successful white- and black-box evasion attacks (Diao et al. 2024), so a detector that cannot keep improving after deployment is already obsolete.

![](images/e2936f25f009dd2c011b55cd4bc1b09cb4d83e6dca30e28def898d027b4e7163.jpg)

(c) Accuracy rises every round  
![](images/9076035ae9c70c4b2432502591e8f843fd2dfed2fb856eed5e149a3e1cf91531.jpg)  
Figure 1: (a) A detector trained once on a static corpus is a stationary target for generators that keep moving. (b) SPARED instead trains an image-editing attacker against a reasoning detector in alternation: each round, the attacker forges harder paired fakes and the retrained detector becomes a stronger judge. (c) The same 9B backbone, trained in this loop, improves round over round on three external benchmarks; the SFT dip trades accuracy for the reasoning output format, which the adversarial rounds more than recover.

This problem is hard because three generations of detectors have already failed it, for diferent reasons. Traditional signal-, frequency-, and pixel-level detectors (Frank et al. 2020; Wang et al. 2020; Tan et al. 2023) identify upsampling fingerprints, spectral artifacts, or neighboring-pixel irregularities left by a specific family of generator architectures. They can perform well in-distribution but degrade sharply when the fingerprint changes: spectral artifacts difer substantially across generative models, hindering generalization to unseen generators (Karageorgiou et al. 2025). Recent generators such as FLUX (Black Forest Labs 2024) and Stable Difusion 3.5 (Esser et al. 2024; Stability AI 2024) further broaden the synthesis pipelines these detectors must cover. Robust detection therefore also needs semantic evidence: physically implausible geometry, broken object counts, and commonsense violations that require world knowledge rather than pixel-level pattern matching to notice (Tan et al. 2025; Kim et al. 2026).

To capture this semantic residue, a second line of work fine-tunes large pretrained models (CLIP-based detectors (Radford et al. 2021; Ojha, Li, and Lee 2023) or MLLMs) on labeled real/fake data, but it fails along an orthogonal axis. Debiasing eforts (Guillaro et al. 2025; Rajan et al. 2025; Chen et al. 2025) and the sanity-check literature (Yan et al. 2024) have identified a dominant source of inflated generalization numbers. When real and fake images in a training set difer in resolution, JPEG quality, or semantic content (because they were sourced independently rather than paired), the detector learns to key on this dataset-level style gap instead of the generation trace itself. Its accuracy then collapses on any semantically aligned, unbiased test set. The result is a field where reported generalization gains are hard to disentangle from residual dataset bias.

A third, more recent line reframes detection as MLLM reasoning to inherit world knowledge and produce genuine explanations (Zhou et al. 2025; Tan et al. 2025; Huang et al. 2025b; Li et al. 2025; Ji et al. 2026), but two further failure modes emerge here. Supervised fine-tuning on a fixed explanation corpus tends to produce templated, causally shallow rationales rather than an understanding that transfers to unseen artifacts (Zhou et al. 2025); and because the training corpus, however well debiased, is still static, the resulting detector is a stationary target: a fixed decision boundary that the next generator release can be tuned against. The fakes inside a static dataset are also, by construction, only as hard as the detector that existed when the dataset was built. Single-model self-evolution attempts such as ForeAgent (Wu et al. 2026) revise their own reasoning traces on this same fixed data and inherit the same ceiling: no external adversary manufactures new hard cases. The same structural failure, a static corpus permanently behind an adaptive adversary, has already been addressed with attacker-defender training loops in LLM safety (Dai et al. 2025; Wen et al. 2026; Liu et al. 2025) and reasoning (Huang et al. 2025a), and in face-forgery detection with synthesizers that adapt their forgeries to the current detector (Chen et al. 2022; Lin et al. 2024; Chou et al. 2026). That literature also records the loop’s own characteristic failure mode: an adversary rewarded only for winning drifts toward degenerate wins (trivial or of-distribution attacks) unless its reward forces genuine engagement.

We bring this paradigm into visual forgery detection as SPARED (Shortcut-Proof Adversarial Reasoning over Edited Data; Figure 1), with a generative image editor as the adversary rather than a policy over a fixed pool of forgery operations. We find that it also closes the dataset-bias shortcut diagnosed above, as a side efect of how its training data is constructed. Our defender is a reasoning MLLM built on a Qwen3.5-9B backbone (Qwen Team 2026) and trained with accuracy-based GRPO (Shao et al. 2024) to output a free-form reasoning trace followed by an explicit verdict. Because the verdict is the only signal rewarded, explanations cannot score by reproducing templates and improve only insofar as they lead to correct verdicts. This answers the templated-rationale failure above. Our attacker is an RLoptimized image-editing model, a Qwen-Image-Edit-2511 LoRA (Wu et al. 2025) trained with DifusionNFT (Zheng et al. 2025), and it does not synthesize fakes from scratch: it edits real source photographs into their fake counterparts. Because every fake is produced from its own real source image, the two share near-identical content, composition, and resolution, and the only systematic diference left for the defender to learn from is the residual editing trace itself. This closes the content/format shortcut identified above (Guillaro et al. 2025) by construction, rather than by post-hoc alignment.

Closing the dataset-bias shortcut this way only works if the attacker keeps editing in earnest, so the second half of our design is the reward that keeps it honest. The attacker is trained with a gated reward: it is credited only when an instruction-following gate (PaCo) (Ping et al. 2025) confirms the edit was faithfully executed and the resulting pair defeats the current defender. The game therefore cannot be won by skipping the edit or degenerating into of-manifold noise. This gate is the vision-domain analogue of the anti-collapse mechanisms other self-play systems require (Dai et al. 2025; Wen et al. 2026; Huang et al. 2025a). The result is an automatically escalating curriculum of instruction-faithful hard negatives, aimed at the current defender’s blind spot, that no fixed dataset can provide.

The two models are also architecturally heterogeneous, a difusion editor in continuous pixel space against an autoregressive MLLM over discrete tokens, and share no parameters. This is a stronger form of the decoupling that MAGIC (Wen et al. 2026) shows avoids the gradient conflict of shared-parameter self-play, and it contrasts with the closest unified systems in this domain, which couple generation and detection (Zhang et al. 2026) or correction and detection (Xu et al. 2026) inside one backbone. Trained on this self-generated data beyond a lightweight SFT initialization, SPARED improves monotonically over three adversarial iterations across three external benchmarks, two of them fully zero-shot (Tables 1 and 2, Figure 3). On judgestyle detection (Kuckreja et al. 2026) and semantic-anomaly reasoning (Tan et al. 2025) it surpasses every baseline we evaluate, with the single exception of a reasoning model 26 times its size on the former, and its explanation quality rises together with accuracy although the explanation is never rewarded. It also transfers to the ten unseen generator families of Holmes-Set (Zhou et al. 2025), whose fully synthetic images are unlike the locally edited photographs it trains on.

![](images/90e834ce662737761ee45826aeb59557acd932403580074e6e96a6b6a2139e03.jpg)  
Figure 2: Overview of SPARED: one adversarial round, read left to right. Attacker training: from one source photograph and its edit instruction the attacker samples a group of candidate edits, and the gated reward credits a rollout only when the instruction was faithfully executed and the result fools the frozen defender, so fooling without editing earns nothing (lock 2). Pool regeneration: the trained attacker re-edits a fixed set of real photographs into pairs whose two halves difer only by the edit (lock 3); the pair shown is an actual training pair. Defender training: both halves of a pair are supervised, and among the sampled responses only verdict correctness is rewarded, so explanations cannot be gamed (lock 1). The retrained defender’s frozen snapshot becomes the judge for round t+1.

Our contributions are:

• A decoupled attacker–defender reinforcement-learning loop for visual forgery detection whose adversary is a generative image editor rather than a policy over a fixed pool of forgery operations, showing that editing real images into semantically aligned fakes closes this field’s dataset-bias shortcut at the training-data level, rather than through post-hoc alignment.

• A heterogeneous, continuous-pixel-space adversary (a PaCo-gated, RL-trained difusion image editor) paired with a decoupled reasoning-MLLM defender, extending the anti-collapse mechanisms of prior self-play systems to the vision domain.

• Empirical evidence that adversarially regenerated edited data alone induces monotonic generalization across three external benchmarks (two of them zero-shot) spanning judge-style detection, semantic-anomaly reasoning, and ten unseen image generators, together with an explicit account of where this signal regresses.

## Method

## Overview

Our pipeline instantiates the attacker-defender loop introduced in the Introduction (Figure 2) as two independently parameterized models trained in alternation: a reasoning defender $( \pi _ { D } )$ that maps an image to a verdict and an explanation, and an image-editing attacker (π ) that maps a real photograph and an edit instruction to a synthetic counterpart of that same photograph. Round t trains $\pi _ { A }$ against a frozen snapshot of $\pi _ { D } ^ { ( t ) }$ to produce a new pool of hard-negative edited images, and the next round trains π<sub>D</sub> on that same pool; the two models never share parameters or a training objective, only this exchanged data and reward signal.

Every design decision below follows a single principle: in an adversarial training loop, any reward or data channel that can be satisfied by a shortcut will be, so the loop escalates genuine capability only if every channel connecting the two models is shortcut-proof. The loop has exactly three such channels, each admitting one of the failure modes diagnosed in the Introduction, and each subsection below closes one: the defender’s reward credits only the final verdict, so explanations cannot be gamed independently of detection (Reasoning Defender); the attacker’s reward is gated on instruction fidelity, so fooling without genuinely editing earns nothing (PaCo-Gated Difusion Attacker); and every fake is paired with its own real source under strict deduplication, so provenance cannot substitute for the editing trace (Data Construction and Training Schedule). With all three channels closed, escalation can occur only through capability: the data gets harder only if the attacker genuinely improves, and the defender gains only by defeating data constructed to defeat it.

## Reasoning Defender

On the defender’s channel, the shortcut is the explanation itself: a detector supervised to reproduce fixed explanation text can satisfy its objective by imitating the phrasing of its training rationales, without ever locating the causal evidence behind a verdict. We therefore leave the defender’s reasoning trace free-form and optimize only the final verdict, so that any reasoning strategy the model discovers is retained solely because it correlates with getting the label right.

The defender is built on a Qwen3.5-9B multimodal backbone (Qwen Team 2026). Given an image x, it generates a response containing an openended <reasoning> $\ldots < /$ reasoning> span followed by <answer>yˆ</answer>, where $\hat { y } \in \{ \mathrm { r e a l } , \mathrm { f a k e } \}$ . We first LoRA-tune (Hu et al. 2022) the base model on real/fake reasoning pairs from the judge-training corpus released with DeepfakeJudge (Kuckreja et al. 2026), disjoint from its evaluation split, to teach the tag format and elementary artifact vocabulary, then merge the adapter and continue with fullparameter GRPO (Shao et al. 2024). The corpus size and the adapter and optimization hyperparameters are listed in the technical appendix. For each training image x with groundtruth label $y ,$ we sample a group of G responses and assign

$$
r _ { D } ( x , y ) = \mathcal { H } [ \hat { y } = y ] ,\tag{1}
$$

crediting nothing for an unparsable answer or a wrong label; GRPO normalizes $r _ { D }$ within the group to form the policygradient advantage, with no separate reward term for format, length, or reasoning content.

Because $r _ { D }$ never rewards the reasoning span directly, reasoning is reinforced only through the correct verdicts it leads to: explanation quality becomes an emergent side efect of accuracy rather than an independent target that can be gamed on its own.

## PaCo-Gated Difusion Attacker

On the attacker’s channel, the shortcut is fooling without editing: were $\pi _ { A }$ rewarded purely for defeating $\pi _ { D }$ , the optimum would be degenerate, since leaving the image essentially unedited and pushing it of the manifold of plausible photographs both fool a detector without producing a meaningful hard negative. Closing this shortcut means making a faithfully executed edit a precondition for any fooling credit.

$\pi _ { A }$ is a LoRA adapter (Hu et al. 2022) on Qwen-Image-Edit-2511 (Wu et al. 2025), trained online with DifusionNFT (Zheng et al. 2025). Given a real source photograph $x _ { \mathrm { s r c } }$ and an edit instruction $c ,$ both drawn from a mixed single-turn editing corpus (ImgEdit, pico-banana-400k, MagicBrush) (Ye et al. 2025; Qian et al. 2025; Zhang et al. 2023), $\pi _ { A }$ produces an edited image $x _ { \mathrm { e d i t } } = \pi _ { A } ( x _ { \mathrm { s r c } } , c )$ paired with its own source as $( x _ { \mathrm { { s r c } } } , \mathrm { { r e a l } ) }$ and $( x _ { \mathrm { { e d i t } } } , \mathrm { { f a k e } ) }$ Every rollout is scored by two frozen judges: an instructionfollowing scorer $r _ { \mathrm { P a C o } } ( x _ { \mathrm { s r c } } , c , x _ { \mathrm { e d i t } } ) ~ \in ~ [ 0 , 1 ]$ (Ping et al. 2025), and the current defender snapshot $\pi _ { D } ^ { ( t ) }$ , queried on both images in the pair to produce a reversed detection reward

$$
r _ { \mathrm { d e t } } = \mathcal { k } ^ { 2 } \big [ \neg ( \pi _ { D } ^ { ( t ) } ( x _ { \mathrm { s r c } } ) = \mathrm { r e a l } \ \wedge \ \pi _ { D } ^ { ( t ) } ( x _ { \mathrm { e d i t } } ) = \mathrm { f a k e } ) \big ] ,\tag{2}
$$

i.e., the attacker earns nothing when the defender gets the whole pair right. The reward is pair-level: the attacker is also credited when the defender misjudges the untouched source,

which steers the regenerated pool toward the defender’s realside errors as well as its fake-side ones. The final reward gates the second on the first:

$$
r _ { A } = { \left\{ \begin{array} { l l } { r _ { \mathrm { d e t } } } & { { \mathrm { i f ~ } } r _ { \mathrm { P a C o } } \geq 0 . 7 , } \\ { 0 } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{3}
$$

optimized with DifusionNFT.

The gate makes $r _ { \mathrm { d e t } } = 1$ a necessary but not suficient condition for reward: an edit that fools $\dot { \pi } _ { D } ^ { ( t ) }$ by ignoring the instruction still scores zero, so only rollouts that are simultaneously instruction-faithful and adversarial to the defender’s current boundary receive positive reward. Because $\pi _ { D } ^ { ( t ) }$ is refreshed every round, the region of image space that satisfies both constraints moves with the defender, so each round’s data pool arrives at a dificulty level calibrated to the model it will retrain.

## Data Construction and Training Schedule

The third channel is the training pool itself. Its shortcut, separating reals from fakes by provenance rather than by editing trace, is a property of how the pool is assembled rather than of either model, so it must be closed at the dataconstruction stage. Every fake in every round must come from editing a real image that is also present as that pair’s real label, and no source image may leak into the images used to build our evaluation benchmarks.

Both the real sources and the edit instructions come from the mixed single-turn editing corpus introduced above. A fixed source set is randomly sampled from this corpus once, deduplicated, and screened by perceptual-hash matching against every benchmark used in our subsequent evaluation, so that no training source leaks into any reported result. Each round’s pool then re-edits this same source set with the current attacker; corpus filtering rules and screening thresholds are given in the technical appendix.

The resulting schedule alternates the two models over five rounds. Starting from the LoRA-SFT checkpoint merged into a full model, the first defender GRPO round (Iter1) trains on a pool edited by the base image-editing model. Each attacker round (A1, A2) then trains $\pi _ { A }$ against a frozen snapshot ofthe preceding defender, and the next defender round (Iter2, Iter3) continues GRPO from the previous defender checkpoint on a pool regenerated by the newly trained attacker; Iter3 is the defender reported as SPARED throughout this paper.

Alternating rather than jointly optimizing the two models keeps each optimization well-posed: $\pi _ { A }$ needs a judge that is stationary within an RL run, and $\pi _ { D }$ needs a pool that does not shift mid-run.

## Experiments

## Experimental Setup

We evaluate SPARED on three externally curated benchmarks: DeepfakeJudge-Detect (Kuckreja et al. 2026), a judge-style benchmark whose fake pool combines text-toimage generations with locally edited real photographs; AnomReason-Deepfake (Tan et al. 2025), which scores both the verdict and the semantic quality of the accompanying explanation through classification-aware semantic AP (CSemAP); and Holmes-Set (Zhou et al. 2025), fully synthetic images from ten generator families that never appear in our training data. We anchor the evaluation on the first two, which are recent, far from saturated, and hard for diferent reasons: DeepfakeJudge-Detect draws its fakes from leaderboard-frontier commercial editors (Nano-Banana, SeedDream, Flux-Kontext, Qwen-Edit), while AnomReason-Deepfake screens its pool (Midjourney, SD3.5, FLUX) for photorealism and scores the explanation alongside the verdict. Holmes-Set is the mature fullysynthetic task and serves as an out-of-genre transfer probe.

<table><tr><td>Model</td><td>Real Acc</td><td>Real F1</td><td>Fake Acc</td><td>Fake F1</td><td>Overall Acc</td><td>Overall F1</td></tr><tr><td colspan="7">Open</td></tr><tr><td>InternVL3.5-1B-HF (Wang et al. 2025)</td><td>47.8</td><td>63.0</td><td>47.8</td><td>11.4</td><td>47.8</td><td>37.2</td></tr><tr><td>Qwen3-VL-2B-Instruct (Bai et al. 2025a)</td><td>49.8</td><td>44.7</td><td>49.8</td><td>54.0</td><td>49.8</td><td>49.4</td></tr><tr><td>Qwen3-VL-8B-Instruct (Bai et al. 2025a)</td><td>50.4</td><td>23.7</td><td>50.4</td><td>63.2</td><td>50.4</td><td>43.5</td></tr><tr><td>Google-Gemma-12B (Gemma Team et al. 2025)</td><td>57.7</td><td>57.4</td><td>49.4</td><td>54.0</td><td>53.6</td><td>55.7</td></tr><tr><td>Microsoft-Phi-4-Instruct (Abdin et al. 2024b)</td><td>61.0</td><td>60.8</td><td>54.4</td><td>58.2</td><td>57.7</td><td>59.5</td></tr><tr><td>InternVL3.5-GPT-OSS-20B-A4B (Wang et al. 2025)</td><td>55.6</td><td>67.6</td><td>55.6</td><td>29.2</td><td>55.6</td><td>48.4</td></tr><tr><td>Qwen3-VL-30B (Bai et al. 2025a)</td><td>94.6</td><td>74.5</td><td>41.0</td><td>56.0</td><td>67.8</td><td>65.3</td></tr><tr><td>Qwen3-VL-235B (Bai et al. 2025a)</td><td>93.5</td><td>78.6</td><td>55.4</td><td>68.4</td><td>74.5</td><td>73.5</td></tr><tr><td colspan="7">Closed</td></tr><tr><td>Gemini-2.5-Flash (Comanici et al. 2025)</td><td>96.6</td><td>73.7</td><td>34.5</td><td></td><td>65.6</td><td>61.9</td></tr><tr><td>ChatGPT-4o-mini (OpenAI 2024a)</td><td>95.8</td><td>70.2</td><td>22.7</td><td>5.</td><td>59.3</td><td>53.0</td></tr><tr><td colspan="7">Reasoning</td></tr><tr><td>Qwen3-VL-8B-Thinking (Bai et al. 2025a)</td><td>67.1</td><td>67.1</td><td>78.7</td><td>69.9</td><td>72.9</td><td>68.5</td></tr><tr><td>Qwen3-VL-30B-Thinking (Bai et al. 2025a)</td><td>67.4</td><td>66.0</td><td>87.6</td><td>72.9</td><td>77.5</td><td>69.5</td></tr><tr><td>Qwen3-VL-235B-Thinking (Bai et al. 2025a)</td><td>75.0</td><td>76.6</td><td>90.3</td><td>79.8</td><td>82.7</td><td>78.2</td></tr><tr><td colspan="7">Deepfake-Specialized</td></tr><tr><td>SIDA-13B-Description (Huang et al. 2025c)</td><td>67.6</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-Gen-Buster++ (Wen et al. 2025a)</td><td>49.9</td><td>57.0 40.0</td><td>27.9 49.9</td><td>34.5 66.5</td><td>47.8 49.9</td><td>45.8 53.3</td></tr><tr><td>UniGenDet (Zhang et al. 2026)</td><td>0.0</td><td>0.0</td><td>99.9</td><td>66.6</td><td>50.0</td><td>33.3</td></tr><tr><td colspan="7">Ours</td></tr><tr><td>Qwen3.5-9B (Qwen Team 2026)</td><td>65.7</td><td>68.7</td><td>65.0</td><td>72.1</td><td>65.4</td><td>70.4</td></tr><tr><td>Qwen3.5-9B-LoRA-SFT (Qwen Team 2026; Hu et al. 2022)</td><td>57.1</td><td>65.4</td><td>82.0</td><td>73.0</td><td>69.6</td><td>69.2</td></tr><tr><td>Qwen3.5-9B-Iter1 (Qwen Team 2026)</td><td>59.3</td><td>68.2</td><td>84.9</td><td>75.3</td><td>72.1</td><td>71.8</td></tr><tr><td>Qwen3.5-9B-Iter2 (Qwen Team 2026)</td><td>66.4</td><td>73.5</td><td>85.5</td><td>78.1</td><td>76.0</td><td>75.8</td></tr><tr><td>Qwen3.5-9B-Iter3 (Qwen Team 2026)</td><td>70.2</td><td>77.4</td><td>88.8</td><td>81.2</td><td>79.5</td><td>79.3</td></tr></table>

Table 1: Comparison of SOTA open-source, closed-source, reasoning, and deepfake models on the DeepfakeJudge-Detect dataset (Kuckreja et al. 2026). Invalid or unparsable answers are counted as incorrect; Real Acc and Fake Acc are per-class recall, and Overall F1 is the macro average of real and fake F1. Red bold and blue underline mark the best and second-best overall scores; the shaded row is our final model.

<table><tr><td>Model</td><td>Acc</td><td>CSemAP-Phe</td><td>CSemAP-Rea</td><td>CSemAP-Full</td></tr><tr><td colspan="5">Open</td></tr><tr><td>LLaVA-OV-7B (Li et al. 2024)</td><td>66.26</td><td>0.1235</td><td>0.1124</td><td>0.1141</td></tr><tr><td>Phi-3.5-Vision (Abdin et al. 2024a)</td><td>41.33</td><td>0.0685</td><td>0.0602</td><td>0.0616</td></tr><tr><td>InternVL2.5-8B (Chen et al. 2024)</td><td>63.61</td><td>0.1165</td><td>0.1085</td><td>0.1089</td></tr><tr><td>InternVL3-8B (Zhu et al. 2025)</td><td>62.84</td><td>0.2949</td><td>0.2394</td><td>0.2559</td></tr><tr><td>InternVL3-9B (Zhu et al. 2025)</td><td>64.04</td><td>0.2293</td><td>0.1987</td><td>0.2081</td></tr><tr><td>Qwen2-VL-7B (Wang et al. 2024)</td><td>67.21</td><td>0.1710</td><td>0.1421</td><td>0.1496</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al. 2025b)</td><td>65.41</td><td>0.1295</td><td>0.1085</td><td>0.1155</td></tr><tr><td>Qwen2.5-VL-72B (Bai et al. 2025b)</td><td>77.60</td><td>0.2626</td><td>0.2337</td><td>0.2453</td></tr><tr><td colspan="5">Closed</td></tr><tr><td>Gemini-2.5-Pro (Comanici et al. 2025)</td><td>85.65</td><td>0.2631</td><td>0.2192</td><td>0.2382</td></tr><tr><td>GPT-o3 (OpenAI 2025b)</td><td>85.60</td><td>0.3189</td><td>0.2690</td><td>0.2898</td></tr><tr><td>GPT-5 (OpenAI 2025a)</td><td>75.22</td><td>0.1790</td><td>0.1535</td><td>0.1658</td></tr><tr><td>GPT-4o (OpenAI 2024b)</td><td>87.76</td><td>0.3750</td><td>0.3487</td><td>0.3612</td></tr><tr><td colspan="5">Deepfake-Specialized</td></tr><tr><td>AnomReasonor-7B (Tan et al. 2025)</td><td>82.61</td><td>0.3684</td><td>0.3574</td><td>0.3613</td></tr><tr><td>UniGenDet (Zhang et al. 2026)</td><td>88.59</td><td>0.4729</td><td>0.3965</td><td>0.4234</td></tr><tr><td colspan="5">Ours</td></tr><tr><td>Qwen3.5-9B (Qwen Team 2026)</td><td>84.05</td><td>0.3268</td><td>0.2896</td><td>0.3063</td></tr><tr><td>Qwen3.5-9B-LoRA-SFT (Qwen Team 2026; Hu et al. 2022)</td><td>77.83</td><td>0.5029</td><td>0.4794</td><td>0.4792</td></tr><tr><td>Qwen3.5-9B-Iter1 (Qwen Team 2026)</td><td>85.72</td><td>0.5164</td><td>0.4909</td><td>0.4927</td></tr><tr><td>Qwen3.5-9B-Iter2 (Qwen Team 2026)</td><td>87.58</td><td>0.5349</td><td>0.5065</td><td>0.5103</td></tr><tr><td>Qwen3.5-9B-Iter3 (Qwen Team 2026)</td><td>92.18</td><td>0.5468</td><td>0.5167</td><td>0.5207</td></tr></table>

Table 2: Deepfake detection with explanation scoring on AnomReason-Deepfake (Tan et al. 2025). Acc measures binary deepfake classification, while CSemAP reports classification-aware semantic quality for phenomenon (Phe), reasoning (Rea), and full matching. Red bold and blue underline mark the best and second-best values per column; the shaded row is our final model.

Baseline pools are benchmark-specific: each table compares against the strongest available systems of each kind, while UniGenDet and every SPARED stage appear in all three as a common reference. AnomReason-Deepfake and Holmes-Set are fully zero-shot; for DeepfakeJudge-Detect, only the SFT initialization uses the suite’s designated training corpus. GenShield, which has no public release, appears only in the Holmes-Set comparison. Training configurations are listed in the technical appendix.

## Main Results

Judge-style detection (Table 1). Every adversarial round improves every aggregate metric on the primary benchmark: overall accuracy rises $6 9 . 6  7 2 . 1  7 6 . 0  7 9 . 5$ from SFT through Iter3, with real recall (57.1 → 70.2) and fake recall (82.0 → 88.8) climbing together rather than trading of. At 9B parameters, the final model surpasses every nonreasoning MLLM we evaluate, including Qwen3-VL-235B (74.5), and every reasoning model up to 30B (77.5), trailing only Qwen3-VL-235B-Thinking (82.7), a model 26× its size. The per-class columns show where this margin comes from: strong closed models answer “real” almost unconditionally (96.6/95.8 real recall against 34.5/22.7 fake recall), while the synthetic-image specialist UniGenDet defaults the other way and answers “fake” on essentially every image (0.0/99.9); the same classification head reaches 99.2 on Holmes-Set (supplementary material), so its collapse here measures distribution shift, not a weak detector. Most open baselines sit near chance on one or both classes. SPARED and the 26×-larger 235B-Thinking are the only two models holding both recalls above 70. The fake pool also splits by construction into a text-to-image and a locally edited subset. The generated subset is near saturation even for the base model (95.3, and 98.3 after training). The gain concentrates in the edited subset, where a small edit leaves most of the image real: accuracy rises from 44.8 to 82.5 (Figure 3c), matching a training pool built from such edits.

Verdict with reasoning (Table 2). On AnomReason-Deepfake, SPARED reaches 92.18 accuracy zero-shot, above every closed model we compare against (GPT-4o: 87.76), 3.6 points above the strongest baseline, UniGenDet (88.59), and 9.6 points above AnomReasonor (82.61), although the latter is fine-tuned on this benchmark’s training data. Explanation quality moves the same way: CSemAP-Full reaches 0.5207, a 23% relative improvement over the strongest baseline score (0.4234). SFT trades accuracy for the reasoning format (84.05 → 77.83 Acc, but 0.3063 → 0.4792 CSemAP); the subsequent GRPO rounds (Shao et al. 2024), whose reward never scores the explanation, then lift both metrics simultaneously and monotonically (85.72/0.4927 → 87.58/0.5103 → 92.18/0.5207). This is the behavior the Method predicts: once explanations can be reinforced only through the verdicts they lead to, explanation quality rises as a side efect of detection accuracy.

Generalization to unseen generators (Figure 3a; per-generator table in the supplementary material). Holmes-Set is the hardest transfer: the adversarial rounds train exclusively on locally edited photographs, yet must classify fully synthetic images from ten unseen generator families. Mean accuracy rises monotonically from 53.0 (base) through 65.7 (SFT) and 84.5 (Iter1) to 92.8 (Iter3) with zero in-domain data, within 2.8 points of AIGI-Holmes (95.6); Figure 3a shows the expansion per generator. Adding indomain detection data to Iter3, a mix of the FakeClue corpus (Wen et al. 2025b) and the benchmark’s own training set (Iter3+Holmes), reaches 97.2 mean accuracy and 99.9 mean AP, the best AP in the table, closing most of the distance to the strongest specialists, GenShield (98.8) and UniGenDet (99.2). The data alone does not explain this: the same mix given to the SFT model without adversarial rounds (SFT+Holmes) reaches 89.5, behind even the zero-shot Iter3. In-domain data compounds the adversarial rounds rather than substituting for them. This task is also close to saturation, with five detectors above 95, while the two anchored benchmarks still separate models sharply; UniGenDet reaches 99.2 here yet collapses on DeepfakeJudge-Detect, and SPARED holds both regimes (79.5 there, 92.8 zero-shot here).

![](images/053f9b52321c97387611ffe419c9dd7a2e7e8c7112f4bfaa176861f364f84ea4.jpg)  
(c) DFj-Detect accuracy by fake subset

![](images/65c6bdfdbe55cdd6e2e35d59058f9276d5caac4f66d15d3da743842668cfbdbc.jpg)  
Figure 3: (a, b) Per-generator accuracy and AP on Holmes-Set across training stages, with axes ordered by final-round accuracy: each adversarial round expands the accuracy polygon, and the final round’s AP stays above 92 on every family. (c) DeepfakeJudge-Detect accuracy on the two fake subsets: the gain concentrates in the locally edited subset.

Where the signal regresses. The ten-generator breakdown also locates the sharpest regression in the adversarial training signal: Janus accuracy drops from 86.5 (Iter2) to 73.1 (Iter3), even as every other family’s accuracy improves and the mean rises from 90.2 to 92.8. This is not a loss of capacity: with in-domain data added, Janus recovers to 84.3 (Iter3+Holmes). Rather, the hard 0/1 fooling reward has no per-family dificulty control, so the family closest to the region the harder pool no longer exercises can regress even as the mean improves.

## Qualitative Analysis

Figure 4 shows the trained detector on three generated images the base model accepts as real: a wrench whose adjustment worm cannot reach the rack teeth it is supposed to drive, a 1930s street scene holding a modern smartphone, and a classroom poster whose “evaporation” arrow points down into the ocean. Between them they instantiate the semanticanomaly classes the Introduction argued remain detectable once low-level artifacts disappear: implausible geometry, and violations that take historical or scientific world knowledge to notice. In each case the verdict comes with evidence the reader can check against the image, the behavior the verdictonly reward was designed to elicit.

<table><tr><td rowspan="2">Variant</td><td colspan="2">DFJ-Detect</td><td colspan="2">AnomReason-DF</td></tr><tr><td>Acc</td><td>F1</td><td>Acc</td><td>CSemAP</td></tr><tr><td>Iter1 (shared starting point)</td><td>72.1</td><td>71.8</td><td>85.72</td><td>0.4927</td></tr><tr><td>+ static-pool GRPO (4.7× budget)</td><td>74.3</td><td>74.3</td><td>88.00</td><td>0.4998</td></tr><tr><td>+ fresh-source GRPO, frozen attacker (7.1× budget)</td><td>73.7</td><td>73.6</td><td>86.84</td><td>0.4990</td></tr><tr><td>+ one adversarial round, no PaCo gate</td><td>66.7</td><td>63.8</td><td>81.12</td><td>0.4991</td></tr><tr><td>+ one adversarial round, unpaired reals</td><td>73.2</td><td>73.0</td><td>85.18</td><td>0.5005</td></tr><tr><td>+ one adversarial round (Iter2)</td><td>76.0</td><td>75.8</td><td>87.58</td><td>0.5103</td></tr><tr><td>+ adversarial rounds (Iter2 → Iter3)</td><td>79.5</td><td>79.3</td><td>92.18</td><td>0.5207</td></tr></table>

Table 3: Ablation on DeepfakeJudge-Detect (DFJ-Detect) and AnomReason-Deepfake (AnomReason-DF), all variants starting from the same Iter1 model: continued GRPO on the frozen round-one pool (static), on fresh-source pools regenerated by the frozen base editor (no attacker training), one adversarial round with the attacker’s instruction-fidelity gate removed, one round on an unpaired pool (each fake’s paired real replaced by a real from disjoint sources), one gated round (Iter2), and the full schedule (Iter2 → Iter3). Red bold marks the best value per column.

## Ablation Study

Adversarial rounds vs. continued static training (Table 3). If the adversarial rounds merely re-label the benefit of training longer, then freezing the loop after the first round and simply continuing GRPO on the same base-edited pool should match the full schedule. Table 3 shows it does not: the static branch captures only 2.2 of the 7.4 points that the adversarial branch adds on DeepfakeJudge-Detect, and trails on every metric. The trajectory is stronger evidence than the endpoint: the static run peaks early, then declines; real recall collapses from the 60s into the 40s, consistent with a boundary drifting toward “fake”, and explanation quality plateaus (CSemAP-Full never exceeds 0.5120). No static checkpoint approaches Iter3 on detection and explanation jointly, whereas the adversarial rounds lift both monotonically. A second control rules out fresh data alone: regenerating the pool each round from freshly sampled sources with the frozen base editor, from the same Iter1 model, peaks at 73.7 on the primary benchmark and oscillates below that thereafter, despite consuming more total budget than the entire schedule. More optimization does not close the gap, and neither does fresher data; the gains require an attacker that keeps learning to fool the current defender.

![](images/c127f3ac7820fa9acdc111ad149d08e839da579be475d5a5c745e7dac7636754.jpg)  
Figure 4: Qualitative cases. (a) A mechanical violation: the base model judges the image real and explicitly reports no impossible geometry, while SPARED identifies the air gap separating the adjustment worm from the rack teeth and explains why the mechanism could not move the jaw. (b) A temporal violation: the base model accepts a 1930s Shanghai street scene as a genuine historical photograph, while SPARED finds the modern touchscreen smartphone in the woman’s hand and rejects the image as anachronistic. (c) An information-visualization violation: the base model reads the classroom’s water-cycle poster as a standard educational print, while SPARED locates the “evaporation” arrow pointing downward from cloud to ocean and states that the real process runs the other way.

Gating the attacker’s reward (Table 3). We repeat the first attacker round with the gate removed, rewarding fooling alone. At matched budget its behavior drifts as the Method predicts: it passes the instruction-fidelity check 9.6 points less often, its median edit magnitude shrinks by 27%, and it fools the frozen defender slightly more, winning more while editing less. Training a defender round on this pool turns drift into damage: instead of reproducing Iter2’s gain, the round falls below its starting point (66.7 vs. Iter1’s 72.1 on DeepfakeJudge-Detect; 81.12 vs. 85.72 on AnomReason-Deepfake). The failure is diagnostic: real recall collapses to 38.2 while fake recall inflates to 95.1, consistent with a pool whose barely-edited “fakes” teach the defender that realisticlooking images are fake, which is exactly what the gate exists

to prevent.

Unpairing the training pool (Table 3). The last lock is the pairing itself. If the pairing were cosmetic, replacing the reals with photographs from disjoint sources (same fakes, same budget) should not matter. Instead, most of the round’s benefit disappears: the unpaired round gains 1.1 points on DeepfakeJudge-Detect where the paired round (Iter2) gains 3.9, and on AnomReason-Deepfake it lands below even its starting point (85.18 vs. Iter1’s 85.72, against Iter2’s 87.58). Part of the training signal is spent on provenance cues that do not transfer, the dataset-bias shortcut the pairing exists to close.

## Conclusion

We presented SPARED, a decoupled attacker–defender reinforcement-learning loop for reasoning-based AIgenerated image detection: a difusion image editor forges paired fakes under an instruction-fidelity gate, and a reasoning MLLM learns to expose them under a verdict-only reward. Once every channel connecting the two models is shortcut-proof, adversarial pressure converts into capability: the detector improves monotonically across rounds and transfers to three external benchmarks, two of them zeroshot, and neither longer training, fresher data, nor in-domain data substitutes for an evolving attacker. Its main limitation is the hard 0/1 fooling reward: without per-family dificulty control, an isolated per-generator regression persists even as the means rise. A graded dificulty signal is the natural next step.

## References

Abdin, M.; Jacobs, S. A.; Awan, A. A.; et al. 2024a. Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv:2404.14219.

Abdin, M.; et al. 2024b. Phi-4 Technical Report. arXiv:2412.08905.

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; et al. 2025a. Qwen3-VL Technical Report. arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; et al. 2025b. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Black Forest Labs. 2024. FLUX.1 [dev]. Hugging Face model card.

Chen, L.; Zhang, Y.; Song, Y.; Liu, L.; and Wang, J. 2022. Self-Supervised Learning of Adversarial Example: Towards Good Generalizations for Deepfake Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18710–18719.

Chen, R.; Xi, J.; Yan, Z.; Zhang, K.-Y.; Wu, S.; Xie, J.; Chen, X.; Xu, L.; et al. 2025. Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable. InAdvances in Neural Information Processing Systems.

Chen, Z.; Wang, W.; Cao, Y.; et al. 2024. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling. arXiv:2412.05271.

Chou, Y.; Yu, T.; Huang, W.; Zhang, Y.; Dai, T.; and Xia, S.-T. 2026. Improving Deepfake Detection with Reinforcement Learning-Based Adaptive Data Augmentation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 3381–3389.

Comanici, G.; Bieber, E.; Schaekermann, M.; et al. 2025. Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities. arXiv:2507.06261.

Dai, M.; Liu, S.; Zhao, Z.; Gao, J.; Sun, H.; and Li, X. 2025. Secure Tug-of-War (SecTOW): Iterative Defense-Attack Training with Reinforcement Learning for Multimodal Model Security. 11414–11423. ACM.

Diao, Y.; Zhai, N.; Miao, C.; Yu, Z.; Wei, X.; Yang, X.; and Wang, M. 2024. Vulnerabilities in AI-Generated Image Detection: The Challenge of Adversarial Attacks. arXiv:2407.20836.

Esser, P.; Kulal, S.; Blattmann, A.; Entezari, R.; Müller, J.; Saini, H.; Levi, Y.; Lorenz, D.; et al. 2024. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis. In International Conference on Machine Learning.

Frank, J.; Eisenhofer, T.; Schönherr, L.; Fischer, A.; Kolossa, D.; and Holz, T. 2020. Leveraging Frequency Analysis for Deep Fake Image Recognition. International Conference on Machine Learning.

Gemma Team; et al. 2025. Gemma 3 Technical Report. arXiv:2503.19786.

Guillaro, F.; Zingarini, G.; Usman, B.; Sud, A.; Cozzolino, D.; and Verdoliva, L. 2025. A Bias-Free Training Paradigm for More General AI-generated Image Detection. 18685– 18694. IEEE.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

Huang, C.; Yu, W.; Wang, X.; Zhang, H.; Li, Z.; Li, R.; Huang, J.; Mi, H.; et al. 2025a. R-Zero: Self-Evolving Reasoning LLM from Zero Data. arXiv.org.

Huang, O.; Lin, M.; Tan, J.; Du, X.; Qiu, Y.; Zheng, J.; Kong, X.; Jiang, Y.; et al. 2026. MIRAGE: Towards AI-Generated Image Detection in the Wild. In Proceedings of the AAAI Conference onArtificial Intelligence, volume 40, 5076–5084.

Huang, T.-M.; Lin, W.-T.; Hua, K.-L.; Cheng, W.-H.; Yamagishi, J.; and Chen, J.-C. 2025b. ThinkFake: Reasoning in Multimodal Large Language Models for AI-Generated Image Detection. arXiv:2509.19841.

Huang, Z.; Hu, J.; Li, X.; He, Y.; Zhao, X.; Peng, B.; Wu, B.; Huang, X.; et al. 2025c. SIDA: Social Media Image Deepfake Detection, Localization and Explanation with Large Multimodal Model. 28831–28841. IEEE.

Ji, Y.; Hong, Y.; Fan, Q.; Lan, J.; Zhu, H.; Wang, W.; Zhang, L.; and Zhang, J. 2026. FakeXplain: AI-Generated Image Detection via Human-Aligned Grounded Reasoning. In International Conference on Learning Representations.

Karageorgiou, D.; Papadopoulos, S.; Kompatsiaris, I.; and Gavves, E. 2025. Any-Resolution AI-Generated Image Detection by Spectral Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18706–18717.

Kim, J.; Kim, S.; Lee, S.; Jung, S.; Rehg, J. M.; and Choi, H. 2026. When Eyes Betray AI: Social Gaze Consistency as a Semantic Cue for AI-Generated Image Detection. arXiv:2605.27348.

Kuckreja, K.; Gupta, P.; Khan, M. H.; and Dhall, A. 2026. Pixels Don’t Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision.

Li, B.; et al. 2024. LLaVA-OneVision: Easy Visual Task Transfer. arXiv:2408.03326.

Li, T.; Huang, Z.; Wen, H.; He, Y.; Lyu, S.; Wu, B.; and Cheng, G. 2025. RAIDX: A Retrieval-Augmented Generation and GRPO Reinforcement Learning Framework for Explainable Deepfake Detection. In Proceedings ofthe 33rd ACM International Conference on Multimedia. ACM.

Lin, Y.; Song, W.; Li, B.; Li, Y.; Ni, J.; Chen, H.; and Li, Q. 2024. Fake It till You Make It: Curricular Dynamic Forgery Augmentations towards General Deepfake Detection. In European Conference on Computer Vision.

Liu, M.; Jiang, L.; Liang, Y.; Du, S. S.; Choi, Y.; Althof, T.; and Jaques, N. 2025. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models. arXiv.org.

Ojha, U.; Li, Y.; and Lee, Y. J. 2023. Towards Universal Fake Image Detectors that Generalize Across Generative Models. 24480–24489. IEEE.

OpenAI. 2024a. GPT-4o mini: Advancing Cost-Eficient Intelligence. OpenAI release blog.

OpenAI. 2024b. GPT-4o System Card.

OpenAI. 2025a. Introducing GPT-5.

OpenAI. 2025b. OpenAI o3 and o4-mini System Card.

Park, J.; and Owens, A. 2025. Community Forensics: Using Thousands of Generators to Train Fake Image Detectors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 8245–8257.

Ping, B.; Jia, C.; Luo, M.; Xia, C.; Shen, X.; Dang, Z.; and Qian, H. 2025. PaCo-RL: Advancing Reinforcement Learning for Consistent Image Generation with Pairwise Reward Modeling. arXiv:2512.04784.

Qian, Y.; Bocek-Rivele, E.; Song, L.; Tong, J.; Yang, Y.; Lu, J.; Hu, W.; and Gan, Z. 2025. Pico-Banana-400K: A Large-Scale Dataset for Text-Guided Image Editing. arXiv:2510.19808.

Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Meila, M.; and Zhang, T., eds., Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 8748– 8763. PMLR.

Rajan, A. S.; Ojha, U.; Schloesser, J.; and Lee, Y. J. 2025. Aligned Datasets Improve Detection of Latent Difusion-Generated Images. In International Conference on Learning Representations.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; et al. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300.

Stability AI. 2024. Stable Difusion 3.5 Large. Hugging Face model card.

Tan, C.; Liu, H.; Zhao, Y.; Wei, S.; Gu, G.; Liu, P.; and Wei, Y. 2023. Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection. arXiv.org.

Tan, C.; Ming, X.; Wang, J.; Tao, R.; Li, B.; Wei, Y.; Zhao, Y.; and Lu, Y. 2025. Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images. arXiv.org.

Wang, P.; Bai, S.; Tan, S.; et al. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv:2409.12191.

Wang, S.-Y.; Wang, O.; Zhang, R.; Owens, A.; and Efros, A. A. 2020. CNN-Generated Images Are Surprisingly Easy to Spot... for Now. 8692–8701. IEEE.

Wang, W.; Gao, Z.; Gu, L.; et al. 2025. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Eficiency. arXiv:2508.18265.

Wen, H.; Li, T.; Huang, Z.; He, Y.; and Cheng, G. 2025a. BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM. arXiv:2507.14632.

Wen, S.; Ye, J.; Feng, P.; Kang, H.; Wen, Z.; Chen, Y.; Wu, J.; Wu, W.; et al. 2025b. Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation. arXiv.org.

Wen, X.; He, Z.; Qi, H.; Wan, Z.; Ma, Z.; Wen, Y.; Zheng, T.; Xu, X.; et al. 2026. MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety. arXiv.org.

Wu, C.; Li, J.; Zhou, J.; Lin, J.; Gao, K.; Yan, K.; ming Yin, S.; Bai, S.; et al. 2025. Qwen-Image Technical Report. arXiv:2508.02324.

Wu, Y.; Yan, K.; Liu, Y.; Zhou, J.; Huang, F.; Zhang, R.; Zhao, Z.; and Wu, F. 2026. Perception, Verdict, and Evolution: Hindsight-Driven Self-Refining Forensics Agent for AI-Generated Image Detection.

Xu, Z.; Zhang, X.; Xu, Y.; Huang, Q.; Chen, S.; Yao, T.; Ding, S.; and Zhang, J. 2026. GenShield: Unified Detection and Artifact Correction for AI-Generated Images.

Yan, S.; Li, O.; Cai, J.; Hao, Y.; Jiang, X.; Hu, Y.; and Xie, W. 2024. A Sanity Check for AI-generated Image Detection. International Conference on Learning Representations.

Ye, Y.; He, X.; Li, Z.; Lin, B.; Yuan, S.; Yan, Z.; Hou, B.; and Yuan, L. 2025. ImgEdit: A Unified Image Editing Dataset and Benchmark. arXiv:2505.20275.

Zhang, K.; Mo, L.; Chen, W.; Sun, H.; and Su, Y. 2023. MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing. In Advances in Neural Information Processing Systems.

Zhang, Y.; Zheng, W.; Li, Y.; Yu, B.; Zheng, Y.; Chen, L.; Lu, J.; and Zhou, J. 2026. UniGenDet: A Unified Generative-Discriminative Framework for Co-Evolutionary Image Generation and Generated Image Detection.

Zheng, K.; Chen, H.; Ye, H.; Wang, H.; Zhang, Q.; Jiang, K.; Su, H.; Ermon, S.; et al. 2025. DifusionNFT: Online Diffusion Reinforcement with Forward Process. arXiv preprint arXiv:2509.16117.

Zhou, Z.; Luo, Y.; Wu, Y.; Sun, K.; Ji, J.; Yan, K.; Ding, S.; Sun, X.; et al. 2025. Aigi-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models. 18746–18758. IEEE.

Zhu, J.; Wang, W.; Chen, Z.; et al. 2025. InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv:2504.10479.