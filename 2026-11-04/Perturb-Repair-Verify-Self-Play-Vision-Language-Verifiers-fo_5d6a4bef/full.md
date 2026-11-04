# Perturb, Repair, Verify: Self-Play Vision-Language Verifiers for Compositional Understanding

Zijun Huang<sup>1</sup> Yingjun Du<sup>2∗</sup> Cees Snoek<sup>2</sup>

<sup>1</sup>University of Southern California <sup>2</sup>VisLab, University of Amsterdam

## Abstract

Vision-language verifiers judge whether an image faithfully matches a textual description, yet existing training pipelines typically rely on static supervision from stronger external judges or curated hard-negative data. Such supervision requires external generation or annotation, is teacher-bounded, and is difficult to adapt as model failures evolve. We introduce PROVE (Perturb-and-Repair Optimization for Vision-Language Evaluation), a self-play framework that trains verifiers through a closed perturb–verify–repair–verify loop. Starting from a known-correct imagecaption pair, a Mutator introduces a typed perturbation over objects, attributes, counts, or relations. The verifier then judges the perturbed caption and provides textual feedback, which a Repairer uses to recover a faithful caption for re-verification. These two verification rounds produce structured rewards for informative mismatches and visually grounded repairs. Since perturbation, verification, and repair are role-conditioned behaviors of the same evolving multimodal backbone, updating the shared parameters also improves the verifier itself. Experiments show that PROVE improves fine-grained compositional verification across model scales and families and strengthens visual-detail robustness on the reference backbone, with limited, benchmark-dependent changes on broader multimodal tasks.

## 1 Introduction

Vision-language models have made rapid progress in image-text understanding, yet they still often miss subtle but consequential mismatches between an image and a textual description. A caption may differ from the image by a substituted object, a misbound attribute, an incorrect count, or a reversed relation, while still retaining high global semantic similarity. As illustrated in Fig. 1a, these errors lie along fine-grained visual-semantic axes that are difficult to capture with coarse image-text scores. This limitation is especially important because VLMs are increasingly used as verifiers: given an image and a description, the verifier judges whether the description faithfully matches the image. Such verifiers support VLM-as-judge evaluation [49], reward modeling for RLHF [29] and RLAIF [16], generated-image filtering, and hallucination detection [9, 19]. A central question is therefore how to train verifiers that recognize localized image-text misalignment, rather than merely score global semantic compatibility.

The dominant recipe imports supervision from outside the target model, as summarized in Fig. 1b. A stronger reference judge, typically a larger VLM or a curated process reward model, scores candidate image-text pairs, and the target verifier is trained to imitate or optimize against that score [20, 39, 46]. Alternatively, training relies on human-labeled or synthetically generated hard negatives [14, 17, 30]. Although effective, both routes provide static supervision. The student is bounded by the perceptual distinctions exposed by the external judge or negative set; the supervision may lag behind evolving model failures; and extending verification to new axes such as counting, relations, or attribute binding often requires new annotation or generation pipelines. The core challenge is to obtain imagetext mismatches that are simultaneously grounded in the image, hard enough to train fine-grained discrimination, and gradable without relying on a stronger external judge.

![](images/318243012520a5e21b137fc420ab0b8f18e85163ca02027ed484e35d38af534e.jpg)  
(a)

![](images/4ecdc0feef61d389614891f6d66e6105b6234cea89e5439ce2606808acb3b3e5.jpg)  
(b)

![](images/a9aa2618daf9216e827ec389655b4516e926608d66839fe867fa6065968dd629.jpg)  
(c)  
Figure 1: Motivation of PROVE. (a) Fine-grained image-text verification requires detecting localized object, attribute, count, and relation mismatches. (b) Existing verifier training relies on static external supervision, while (c) PROVE creates adaptive supervision through a perturb–verify–repair– verify self-play loop.

We introduce PROVE (Perturb-and-Repair Optimization for Vision-Language Evaluation), a selfplay framework that trains a verifier through a closed perturb–verify–repair–verify loop (Fig. 1c). Starting from a known-correct image-caption pair, a Mutator creates a minimally perturbed caption by editing one visual-semantic axis, such as object identity, attribute, count, or relation. The verifier then judges the perturbed caption and provides feedback about the mismatch. Conditioned on this feedback, a Repairer attempts to recover a faithful caption, which is verified again. Crucially, perturbation, verification, and repair are not separate models: they are three role-conditioned behaviors of the same evolving VLM. The two verification rounds produce rewards for informative perturbations and visually grounded repairs, while shared-parameter updates also improve the verifier used at evaluation time.

This formulation is well matched to fine-grained image-text verification. The Mutator does not generate arbitrary text corruptions; it creates controlled, localized mismatches whose semantic axes mirror common compositional failures. The verifier must diagnose the mismatch before repair and recognize the corrected caption after repair, while the Repairer must localize the error and infer the visually faithful description. In this way, PROVE converts ordinary clean image-caption pairs into adaptive hard negatives and repair targets. Compared with external-judge distillation, the training signal is not bounded by a stronger frozen oracle. Compared with fixed hard-negative training, the mismatch distribution evolves with the current verifier.

We evaluate PROVE on a Qwen3.5-2B backbone across fine-grained compositional alignment benchmarks and broader vision-language robustness benchmarks. PROVE consistently improves the same backbone on compositional verification, with the largest gains appearing on axes aligned with the perturbation space, such as attribute and relation understanding. It also improves visual-detail sensitivity and maintains hallucination robustness, while broader capability changes are limited and benchmark-dependent. Training diagnostics further show that perturbation and repair co-adapt through verifier feedback. These results suggest that vision-language verifiers can be strengthened through verifier-guided self-play rather than static judge distillation.

Matched supervision baselines, independent verifier audits, and additional Qwen3.5-4B and InternVL3-2B experiments further support this teacher-free adaptive alternative. The present study establishes caption-based verification and benchmark transfer; task-specific self-play for VQA, grounding, and referring expressions remains future work.

## 2 Related Work

Vision-language verifiers for image-text understanding. Evaluating whether an image faithfully matches a textual description is a central problem in vision-language understanding. Coarse imagetext similarity metrics are often insufficient, because they can miss fine-grained errors in object identity, attribute binding, counting, and spatial relations. Recent work therefore uses vision-language models as verifiers or judges for multimodal evaluation, reward modeling, generated-image filtering, and hallucination detection [8, 9, 19, 49]. These verifiers are attractive because they can provide semantic judgments beyond fixed embedding similarities, but their reliability depends on whether they can detect subtle image-text mismatches. PROVE targets this capability directly by training the verifier through controlled perturbation, verification, and repair over the same compositional axes that commonly challenge image-text understanding.

Training verifiers with static external supervision. A common way to train multimodal verifiers is to import supervision from outside the target model. For example, stronger VLMs or curated reward models can score candidate image-text pairs, after which the target model is optimized by preference learning or reward-based training [18, 20, 37–39, 46]. Another line constructs fixed hard-negative datasets or synthetic preference pairs, then trains the verifier to distinguish faithful captions from controlled corruptions [21]. These approaches provide useful supervision, but the signal is static: it is bounded by the external judge, fixed negative generator, or annotation pipeline. As models and evaluation distributions evolve, their failure modes may shift away from this supervision. In contrast, PROVE does not distill a stronger frozen judge or rely on a fixed hard-negative set. It uses the current verifier itself inside a perturb–verify–repair–verify loop, so the training signal changes with the evolving model.

Self-play and self-evolving multimodal learning. Self-play has been widely used to improve agents and reasoning systems, from game-playing methods such as AlphaZero [36] to debate and recent selfevolving LLMs [12, 13, 47]. More recent asymmetric self-play methods train one behavior to propose tasks and another to solve them, creating an adaptive curriculum without a fixed external teacher [23]. In multimodal learning, self-evolving VLMs have also been explored through questioner–reasoner interactions and pseudo-labeling from model samples [10]. However, consensus- or uncertainty-based supervision can still reinforce confident mistakes, especially when the model fails to perceive the relevant visual evidence. PROVE differs in both task and interaction structure: it trains a visionlanguage verifier rather than a general reasoner, and it grounds self-play in clean image-caption pairs. The Mutator creates typed image-text mismatches, the verifier diagnoses them, and the Repairer attempts to correct them; all three are role-conditioned behaviors of the same evolving MLLM.

Compositional image-text alignment. Vision-language models often struggle with compositional understanding despite strong global image-text alignment. Benchmarks such as Winoground [40], ARO [44], SugarCrepe [11], and VL-CheckList [48] test whether models can distinguish captions that differ in objects, attributes, relations, or word order. Prior methods improve this ability with hard-negative contrastive learning or synthetic data [7, 44]. PROVE is complementary: it generates controlled perturbations online and uses verifier-guided repair to create adaptive supervision for fine-grained image-text alignment.

## 3 Methods

## 3.1 Overview

PROVE trains a vision-language verifier through a closed perturb–verify–repair–verify loop. Given a clean image-caption pair, a single multimodal policy π<sub>θ</sub> is reused under three role prompts: a Mutator that perturbs the caption, a Verifier that judges image-caption faithfulness, and a Repairer that corrects the caption using verifier feedback. These roles are not separate models; they are three behaviors of the same parameter-shared backbone. Thus, every policy update also changes the verifier used at evaluation time. Figure 2 illustrates the overall loop.

The key idea is to turn ordinary positive image-caption pairs into adaptive verification episodes. The Mutator creates localized mismatches along object, attribute, count, or relation axes. The Verifier diagnoses whether the perturbed caption still matches the image. The Repairer then attempts to recover a faithful caption from the image and verifier feedback. This interaction produces grounded hard negatives and repair targets without a stronger external judge or a fixed hard-negative dataset.

![](images/a7379455b3e9fa543328f9e6e102cb36225be597f9688af610c921fde01f6a44.jpg)  
Figure 2: Overview of PROVE. A single Qwen backbone is reused under three role prompts: Mutator, Verifier, and Repairer. Given a clean image-caption pair, the Mutator produces a localized caption perturbation, e.g., changing an attribute from “red” to “blue”, to create a hard image-text mismatch. The Verifier first judges the perturbed caption and returns a decision-feedback pair $( y _ { 1 } , r _ { 1 } )$ that diagnoses the mismatch. Conditioned on this feedback, the Repairer generates a revised caption, which is then checked by the same Verifier to obtain $( y _ { 2 } , r _ { 2 } )$ . Rewards from the two verification rounds train the shared model with DrGRPO, so perturbation, verification, and repair co-evolve within one multimodal backbone.

## 3.2 Perturb–Verify–Repair–Verify Self-Play

Let $\mathcal { D } = \{ ( x _ { i } , p _ { i } ) \} _ { i = 1 } ^ { N }$ be a set of image-caption pairs, where $p _ { i }$ faithfully describes image $x _ { i }$ . For a clean pair $( x , p )$ , PROVE constructs the following self-play episode:

$$
( x , p ) \xrightarrow { \mathrm { M u t a t o r } } \ { p ^ { \prime } } \xrightarrow { \mathrm { V e r i f i e r } } \ ( y _ { 1 } , r _ { 1 } ) \xrightarrow { \mathrm { R e p a i r e r } } \ \tilde { p } \xrightarrow { \mathrm { V e r i f i e r } } \ ( y _ { 2 } , r _ { 2 } ) .\tag{1}
$$

Here $p ^ { \prime }$ is a perturbed caption, $( y _ { 1 } , r _ { 1 } )$ is the first verifier decision and explanation, $\tilde { p }$ is the repaired caption, and $( y _ { 2 } , r _ { 2 } )$ is the second verifier decision and explanation. The binary label $y \in \{ 0 , 1 \}$ indicates whether the caption is judged faithful to the image, and r is natural-language feedback.

All stages are sampled from the same policy under different role prompts:

$$
\begin{array} { r } { \pi _ { \theta } ( \cdot  { | \ x , q , z ) , } \qquad z \in \{ \mathbf { M U T A T O R } , \mathbf { V E R I F I E R } , \mathbf { R E P A I R E R } \} , } \end{array}\tag{2}
$$

where $q$ denotes the role-specific text context. This makes the verifier part of the evolving policy, rather than a frozen reward model.

Caption perturbation. The Mutator produces a localized edit

$$
a _ { M } = ( \tau , t , \delta ) ,\tag{3}
$$

where τ is the edit type, t is the target span, and δ specifies the replacement. We use four edit types:

$$
\mathcal { T } = \{ \mathsf { o b j e c t } , \mathsf { a t t r i b u t e } , \mathsf { c o u n t } , \mathsf { r e l a t i o n } \} .\tag{4}
$$

A deterministic editor applies the edit to obtain the perturbed caption:

$$
p ^ { \prime } = E ( p , a _ { M } ) .\tag{5}
$$

Restricting the Mutator to typed local edits prevents arbitrary rewriting and keeps the negative caption close to the original image-caption pair. Invalid edit types, unmatched or ambiguous target spans, unchanged replacements, and non-deterministic edits are filtered; Appendix F describes the role interfaces and editor constraints.

Verification and repair. The Verifier first judges the perturbed caption:

$$
( y _ { 1 } , r _ { 1 } ) \sim \pi _ { \theta } ( \cdot \mid x , p ^ { \prime } , { \mathrm { V E R I F I E R } } ) .\tag{6}
$$

The feedback $r _ { 1 }$ identifies the perceived mismatch and is passed to the Repairer, which generates a corrected caption:

$$
\tilde { p } \sim \pi _ { \theta } ( \cdot \mid x , p ^ { \prime } , y _ { 1 } , r _ { 1 } , { \mathrm { R E P A I R E R } } ) .\tag{7}
$$

The same Verifier then evaluates the repaired caption:

$$
( y _ { 2 } , r _ { 2 } ) \sim \pi _ { \theta } ( \cdot \mid x , \tilde { p } , { \mathrm { \bf V E R I F I E R } } ) .\tag{8}
$$

The first verification tests whether the model detects the perturbation; the second tests whether the repair restores faithfulness. Together, they turn verification into an operational correction task.

## 3.3 Verifier-Guided Rewards

The two verifier calls provide the supervision for self-play. The Mutator should produce perturbations that are informative, grounded, and not overly destructive. The Repairer should recover a faithful caption while making minimal changes.

Mutator reward. We define the Mutator reward as

$$
r _ { M } = R _ { \mathrm { e x p o s e } } ( y _ { 1 } ) + R _ { \mathrm { h a r d } } ( y _ { 2 } ) - \lambda _ { M } C _ { M } ( a _ { M } ) ,\tag{9}
$$

where $R _ { \mathrm { e x p o s e } }$ rewards perturbations that expose verifier blind spots, $R _ { \mathrm { h a r d } }$ rewards perturbations that remain difficult after repair, and $C _ { M }$ penalizes overly wide edits. Invalid actions receive a fixed penalty and do not enter $C _ { M }$ (Appendix F). We instantiate it as

$$
r _ { M } = \lambda _ { 1 } \mathcal { k } [ y _ { 1 } = 1 ] + \lambda _ { 2 } \mathcal { k } [ y _ { 2 } = 0 ] - \lambda _ { 3 } C _ { M } ( a _ { M } ) .\tag{10}
$$

Repairer reward. The Repairer reward encourages a successful and minimal correction:

$$
r _ { R } = R _ { \mathrm { a c c e p t } } ( y _ { 2 } ) + R _ { \mathrm { f l i p } } ( y _ { 1 } , y _ { 2 } ) - \lambda _ { R } C _ { R } ( p ^ { \prime } , \tilde { p } ) ,\tag{11}
$$

where $R _ { \mathrm { a c c e p t } }$ rewards repaired captions accepted by the verifier, $R _ { \mathrm { { f l i p } } }$ rewards changing the verifier judgment, and $C _ { R }$ penalizes unnecessary rewriting. We use

$$
r _ { R } = \beta _ { 1 } \lvert \mathcal { k } [ y _ { 2 } = 1 ] + \beta _ { 2 } \rvert \mathcal { k } [ y _ { 1 } \neq y _ { 2 } ] - \beta _ { 3 } C _ { R } ( p ^ { \prime } , \tilde { p } ) .\tag{12}
$$

Although rewards are assigned to perturbation and repair trajectories, the Verifier is not a frozen judge. Because all roles share θ, updates from these rewards also update the verifier behavior itself. This coupling is what allows PROVE to improve the verifier without distilling a stronger external model.

## 3.4 Role-Conditioned Policy Optimization

We optimize the shared policy with DrGRPO [27]. For each source pair, the model samples multiple self-play trajectories and computes verifier-guided rewards. Let $r _ { i }$ be the reward of trajectory i, and let $\bar { G } ( i )$ be the group of trajectories sampled from the same source context. Following DrGRPO, we use mean-centered advantages without standard-deviation normalization:

$$
\hat { A } _ { i } = r _ { i } - \frac { 1 } { | G ( i ) | } \sum _ { j \in G ( i ) } r _ { j } .\tag{13}
$$

This preserves the reward scale and avoids unstable rescaling when within-group variance is small. The clipped objective over generated tokens is

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { i , t } \left[ \operatorname* { m i n } \left( \rho _ { i , t } ( \theta ) \hat { A } _ { i } , \operatorname { c l i p } \left( \rho _ { i , t } ( \theta ) , 1 - \epsilon ^ { - } , 1 + \epsilon ^ { + } \right) \hat { A } _ { i } \right) \right] ,\tag{14}
$$

where

$$
\rho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( a _ { i , t } \mid q _ { i } , a _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a _ { i , t } \mid q _ { i } , a _ { i , < t } ) } .\tag{15}
$$

Here $q _ { i }$ is the role-conditioned multimodal prompt and $a _ { i , t }$ is the generated token. Equation 14 is a maximization objective; equivalently, its negative is minimized, with KL regularization specified in the experimental setup. At test time, we use the Verifier prompt of the evolved shared model to judge image-caption faithfulness.

## 4 Experiments

## 4.1 Experimental Setup

Backbone and training framework. We instantiate PROVE with Qwen3.5-2B [31] as the shared multimodal backbone. The same parameters are reused across the Mutator, Verifier, and Repairer roles, with role identity specified only by the prompt template. Thus, perturbation, verification, and repair are optimized as role-conditioned behaviors of one evolving MLLM rather than as separate models. At evaluation time, we use the Verifier role of the evolved shared model to judge imagecaption faithfulness.

Training data and self-play construction. We train on a 48k-example subset of ShareGPT-4o-Image-style image-caption pairs [4], with a held-out validation split of 5k examples. Each self-play episode starts from a known-correct image-caption pair. The Mutator introduces a typed perturbation along one of four visual-semantic axes: object, attribute, count, or relation. The Verifier then judges the perturbed caption and produces a decision-feedback pair. Conditioned on this feedback, the Repairer generates a repaired caption, which is judged again by the same Verifier role. The two verification rounds provide verifier-guided rewards for self-play optimization. No stronger external judge, human-labeled hard negatives, or additional hard-negative annotations are used during training.

Optimization and compute. We implement PROVE using the VeRL/HybridFlow training framework [35] and optimize the shared policy with DrGRPO [27], a group-relative policy optimization variant that removes the standard-deviation normalization used in vanilla GRPO [34]. Unless otherwise specified, we use rollout size $n = 3$ , learning rate $5 \times 1 0 ^ { - 7 }$ , global batch size 48, KL coefficient 0.005, and PPO clip range [0.20, 0.22]. All main training runs are conducted on a single node with 8 NVIDIA A100 GPUs using BF16 mixed precision. We report the best validation checkpoint in the main results. Training hyperparameters, role interfaces, and additional experimental results are provided in the appendix. The reference run takes 11 hours on 8 A100 GPUs (88 GPU-hours); Appendix A.4 compares complete pipeline costs. We will release the training and evaluation code to facilitate reproducibility.

Evaluation protocol. Our main evaluation focuses on fine-grained image-text understanding, where a verifier must distinguish captions that differ by localized semantic changes. Table 1 compares PROVE with representative compositional understanding methods on four benchmarks shared with [30]: ARO [44], VL-CheckList [48], ColorSwap [3], and FG-OVD [1]. These benchmarks cover relation and attribute reasoning, object-level grounding, color-word order sensitivity, and fine-grained open-vocabulary recognition. For each baseline, we report the values published in the original paper whenever available; when a benchmark was not reported but public code is available, we use our reproduction under the same evaluation protocol.

Beyond the core compositional suites, we evaluate how the learned verifier transfers to broader vision-language understanding. We include Winoground [40] to test role binding under hard imagetext contrasts, MMVP [41] to assess sensitivity to fine visual details, and MMBench-EN [25] to measure general multimodal perception and reasoning. For all two-alternative forced-choice tasks, we randomize option order for each example to reduce positional bias. Unless otherwise stated, all benchmarks are evaluated using their official protocols and metrics. Comp. Avg. averages the four core benchmark groups in Table 1; Broader Avg. averages Winoground GS, MMVP pair accuracy, and MMBench in Table 2. Winoground GS and the appendix 2-AFC accuracy are distinct metrics. External evaluation labels are not used for training, rewards, mutation generation, validation, or checkpoint selection.

## 4.2 Main Results

Compositional understanding. Table 1 compares PROVE with representative compositional understanding methods on ARO, VL-CheckList, ColorSwap, and FG-OVD. Because these methods differ in backbone architecture, training data, objectives, and benchmark coverage, the cross-method comparison should be interpreted as a contextual reference rather than a fully controlled comparison. Our main controlled comparison is between Qwen3.5-2B and PROVE using the same backbone. Under this controlled setting, PROVE improves the overall average from 83.22 to 86.85, with consistent gains across all four benchmark groups. The largest improvement appears on ARO, where the average score increases by +7.25, indicating stronger attribute and relation discrimination.

Table 1: Comparison with compositional understanding methods. We report all available submetrics on ARO, VL-CheckList, ColorSwap, and FG-OVD. “–” indicates that the benchmark was not evaluated. Best and second-best results are shown in bold and underline. PROVE achieves the best overall average among methods with complete benchmark coverage and consistently improves over the Qwen3.5-2B backbone across all benchmark groups.
<table><tr><td rowspan="2">Model</td><td colspan="3">ARO</td><td colspan="4">VL-CheckList</td><td>ColorSwap</td><td>FG-OVD</td><td>Overall</td></tr><tr><td>Rel</td><td>Attr</td><td>Avg</td><td>Attr</td><td>Obj</td><td>Rel</td><td>Avg</td><td>ITT</td><td>Avg</td><td>Avg</td></tr><tr><td colspan="9">Contextual comparison: discriminative VLMs</td><td></td></tr><tr><td>CLIP [32]</td><td>59.00</td><td>62.00</td><td>60.50</td><td>67.93</td><td>82.83</td><td>64.19</td><td>71.65</td><td>35.67</td><td>47.33</td><td>53.79</td></tr><tr><td>NegCLIP [44]</td><td>81.00</td><td>71.00</td><td>76.00</td><td>72.24</td><td>87.00</td><td>71.39</td><td>76.87</td><td>35.67</td><td>41.69</td><td>57.56</td></tr><tr><td>GNM [33]</td><td>65.00</td><td>65.00</td><td>65.00</td><td>70.15</td><td>85.91</td><td>64.10</td><td>73.38</td><td>13.00</td><td>38.79</td><td>47.54</td></tr><tr><td>PANeg [2]</td><td>65.07</td><td>67.94</td><td>66.51</td><td>76.51</td><td>88.13</td><td>69.90</td><td>78.17</td><td>17.67</td><td>44.98</td><td>51.83</td></tr><tr><td>DAC-SAM [7]</td><td>77.16</td><td>70.50</td><td>73.83</td><td>75.80</td><td>88.50</td><td>89.80</td><td>84.70</td><td>16.33</td><td>48.36</td><td>55.81</td></tr><tr><td>DAC-LLM [7]</td><td>81.28</td><td>73.91</td><td>77.60</td><td>77.30</td><td>87.30</td><td>86.40</td><td>83.66</td><td>18.33</td><td>49.60</td><td>57.30</td></tr><tr><td>XVLM [45]</td><td>73.40</td><td>86.80</td><td>80.10</td><td>75.10</td><td>85.80</td><td>70.40</td><td>76.50</td><td></td><td></td><td></td></tr><tr><td colspan="9">Contextual comparison: generative VLMs</td><td></td><td></td></tr><tr><td>InstructBLIP [6]</td><td>69.20</td><td>50.83</td><td>60.02</td><td>56.37</td><td>80.33</td><td>53.34</td><td>63.35</td><td>40.33</td><td>26.80</td><td>47.63</td></tr><tr><td>MiniGPT-4 [50]</td><td>46.90</td><td>55.70</td><td>51.30</td><td>71.30</td><td>84.20</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BLIP2-VGPS [22]</td><td>90.70</td><td>94.30</td><td>92.50</td><td>73.90</td><td>90.10</td><td>89.90</td><td>84.63</td><td></td><td></td><td></td></tr><tr><td>CapPa [42]</td><td>86.70</td><td>85.70</td><td>86.20</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LLaVA-1.5-7B [24]</td><td>65.08</td><td>57.40</td><td>61.24</td><td>69.49</td><td></td><td>73.50</td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL2-8B [5]</td><td>83.42</td><td>93.74</td><td>88.58</td><td>84.81</td><td></td><td>87.10</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Matched comparison: shared-backbone verifier models</td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-2B [31]</td><td>69.68</td><td>79.48</td><td>74.58</td><td>76.75</td><td>91.20</td><td>82.85</td><td>83.60</td><td>93.67</td><td>81.01</td><td>83.22</td></tr><tr><td>PROVE (Qwen3.5-2B)</td><td>76.92</td><td>86.74</td><td>81.83</td><td>81.46</td><td>94.14</td><td>86.95</td><td>87.52</td><td>95.83</td><td>82.23</td><td>86.85</td></tr></table>

Table 2: Comparison on broader vision-language benchmarks. PROVE achieves the best average and improves the Qwen3.5-2B backbone on all three benchmarks, indicating transfer beyond the core compositional suites. Broader task changes are benchmark-dependent (Appendix C.3).
<table><tr><td>Model</td><td>Wino GS</td><td>MMVP Pair</td><td>MMBench</td><td>Avg</td></tr><tr><td colspan="5">Discriminative VLMs</td></tr><tr><td>CLIP [32]</td><td>8.00</td><td>15.55</td><td></td><td></td></tr><tr><td>NegCLIP [44]</td><td>8.00</td><td>26.67</td><td></td><td></td></tr><tr><td>PANeg [2]</td><td>6.75</td><td>21.48</td><td></td><td></td></tr><tr><td>DAC-SAM [7]</td><td>6.25</td><td>23.70</td><td></td><td></td></tr><tr><td>DAC-LLM [7]</td><td>3.25</td><td>28.14</td><td></td><td></td></tr><tr><td colspan="5">Generative VLMs</td></tr><tr><td>InstructBLIP [6]</td><td>4.75</td><td>14.81</td><td></td><td></td></tr><tr><td>LLaVA-1.5-7B [24]</td><td>27.00</td><td>24.70</td><td>64.30</td><td>38.67</td></tr><tr><td>InternVL2-8B [5]</td><td>49.75</td><td>51.33</td><td>81.70</td><td>60.93</td></tr><tr><td colspan="5">Shared-backbone verifier models</td></tr><tr><td>Qwen3.5-2B [31]</td><td>49.00</td><td>56.00</td><td>71.31</td><td>58.77</td></tr><tr><td>PROVE (Qwen3.5-2B)</td><td>50.75</td><td>60.00</td><td>72.34</td><td>61.03</td></tr></table>

PROVE also improves VL-CheckList, ColorSwap, and FG-OVD, suggesting that the learned verifier benefits not only the perturbation axes seen during self-play but also broader forms of localized image-text grounding. Compared with prior compositional methods, PROVE achieves the best overall average among complete rows in this table and remains competitive on individual benchmark groups, despite using a compact shared backbone and no external hard-negative dataset. These results support the core design of PROVE. The Mutator creates controlled mismatches over objects, attributes, counts, and relations; the Verifier learns to diagnose these mismatches; and the Repairer learns to recover visually faithful captions. The resulting self-play signal strengthens fine-grained image-text verification without relying on a stronger external judge.

Broader vision-language understanding benchmarks. Table 2 evaluates whether the learned verifier transfers beyond the main compositional suites. PROVE improves the Qwen3.5-2B backbone on all three benchmarks, raising the average from 58.77 to 61.03. The most notable gain is on MMVP, where pair accuracy improves by +4.00, suggesting improved sensitivity to fine visual details. Winoground also improves, indicating better role binding under hard image-text contrasts, while MMBench-EN shows a modest positive change, suggesting positive transfer to this general multimodal perception and reasoning benchmark.

Table 3: Matched supervision and core loop ablations. All training variants use Qwen3.5-2B, 48k source pairs, and the same target-model optimization budget and evaluation protocol. Complete ablations are in Appendix B.
<table><tr><td>Method / variant</td><td>Comp. Avg.</td><td>Broader Avg.</td></tr><tr><td>Qwen3.5-2B</td><td>83.22</td><td>58.77</td></tr><tr><td>Fixed typed negatives</td><td>85.02</td><td>59.68</td></tr><tr><td>External-judge distillation</td><td>86.29</td><td>60.72</td></tr><tr><td>Verifier self-training</td><td>85.66</td><td>60.21</td></tr><tr><td>No Repairer or second verification</td><td>84.24</td><td>59.31</td></tr><tr><td>Repairer without verifier feedback</td><td>85.17</td><td>59.84</td></tr><tr><td>Frozen internal Verifier</td><td>85.12</td><td>59.81</td></tr><tr><td>PROVE</td><td>86.85</td><td>61.03</td></tr></table>

(b) Axis-level gains over Qwen3.5-2B

![](images/82f677dab77d185109b10ed48c03a99bceb9cb46e77a033419b8b1c9847283f6.jpg)

![](images/b0c6517e382668af6aa0208e80caba96665d85e6abd6ca1f53b0ad6e822607eb.jpg)

![](images/1c81e2ef1234f8693c50c963e9dc4108a42bcd7e975505072c6cb44bd20a3b13.jpg)

![](images/160d03b75af3be4ecbf8fa12f543db9da6866d5e646bcf15d230072e3f40aaca.jpg)

![](images/4b7e19c841dba43c1a6826ba35e3bef48638797a2a7df2c842bf744b182fd7c0.jpg)

![](images/632292317754a426af55a84fb93d693a9108d3a963cb6174f69abf8ce67f1684.jpg)

![](images/ad7271f97c14680fd25a400b015ddfe0cf2aa841ae4950482243b059cda9fe16.jpg)  
Figure 3: Self-play dynamics and axis-level gains. (a) The Repairer steadily improves in repair success and structural alignment, while the verifier pass rate rises as repairs catch up with the current Mutator distribution; the dashed line marks the lower finite-group maximum-variance reference at 1/3 (Appendix F). (b) Each panel compares Qwen3.5-2B with PROVE on an aggregated visual-semantic capability; Total is the overall average in Table 1. The largest gains appear on attribute and relation verification, matching the typed perturbation axes used during self-play.

Together, Tables 1 and 2 show that PROVE mainly strengthens the verifier on localized image-text discrimination, which matches the intended training signal of typed perturbation and verifier-guided repair. At the same time, the gains on MMVP and MMBench indicate positive transfer, while repeatedrun results reveal small, task-dependent regressions on other broader benchmarks (Appendix C.3).

Matched supervision alternatives. Table 3 compares static negatives generated once, judgments from a frozen larger VLM, and self-generated verifier labels without repair. PROVE improves the compositional/broader averages over these baselines by 1.83/1.35, 0.56/0.31, and 1.19/0.82 points, respectively. These results support adaptive perturb–repair self-play as a teacher-free alternative; its advantage is adaptive supervision rather than uniformly lower compute or universal superiority over distillation.

Table 4: Transfer across model scale and family. The same perturbation and reward formulation improves both evaluation averages.
<table><tr><td rowspan="2">Backbone</td><td colspan="2">Comp. Avg.</td><td colspan="2">Broader Avg.</td></tr><tr><td>Base</td><td>PROVE</td><td>Base</td><td>PROVE</td></tr><tr><td>Qwen3.5-2B</td><td>83.22</td><td>86.85</td><td>58.77</td><td>61.03</td></tr><tr><td>Qwen3.5-4B</td><td>85.15</td><td>88.71</td><td>60.92</td><td>63.26</td></tr><tr><td>InternVL3-2B</td><td>82.19</td><td>85.34</td><td>58.16</td><td>60.99</td></tr></table>

![](images/bab1aa88d0ed95c2898a33215d0ef19f5b31677af5ae866f6df18b613513f575.jpg)  
Figure 4: Qualitative verifier examples. Each panel compares the base Qwen3.5-2B model with PROVE on a fine-grained verification case. The examples cover relation, object existence, attribute, and counting errors. PROVE better localizes the relevant visual evidence and avoids accepting captions or answers that miss the targeted mismatch.

## 4.3 Analysis

Self-Play Dynamics Analysis. Figure 3(a) analyzes how the perturb–verify–repair loop evolves during training. The training dynamics show that the Repairer steadily catches up with the perturbations produced by the Mutator while the overall optimization remains stable. Repair success increases from 36.0% to 62.7%, and structural alignment rises from 61.0% to 78.6%, indicating that the Repairer becomes increasingly capable of localizing Mutator-induced mismatches and recovering visually faithful captions. At the same time, the verifier pass rate increases from 0.360 to 0.633 and remains above the lower finite-group maximum-variance reference point of 1/3, suggesting that the current four-type perturbation space gradually becomes easier for the evolving Repairer to solve. This trend reveals both the effectiveness and the limitation of the current self-play curriculum: typed perturbations provide a useful learning signal, but a richer action vocabulary may be needed once the Repairer becomes strong enough to saturate the existing perturbation space. Finally, the validation reward improves consistently, suggesting that the verifier-guided rewards provide a stable learning signal.

Axis-level gains. Figure 3(b) summarizes where the gains of PROVE come from by aggregating benchmark splits according to the visual-semantic capability they test. The largest improvements appear on attribute and relation verification, which are directly aligned with the typed perturbations used by the Mutator. The full analysis is provided in Appendix G.1.

Controlled loop ablations. Removing the Repairer and second verification lowers the two averages by 2.61/1.72 points; removing verifier feedback costs 1.68/1.19 (Table 3). Freezing the internal Verifier costs 1.73/1.22, supporting verifier co-evolution. Reward and role-sharing ablations also favor the full loop, while replacing DrGRPO with GRPO costs only 0.43/0.32 (Appendix B).

Independent verification. Table 12 shows a 7.8-point verifier accuracy gain and a 23.4-point increase in fully valid trajectories, requiring a real mismatch, correct diagnosis, faithful minimal repair, and correct final judgment. Fixed external benchmark averages also improve by 3.63/2.26 points across training; their correlations with self-play pass rate are 0.9254/0.9022 (Appendix C.2). These independent measurements support improved image-grounded correctness beyond internal role agreement.

Generality and capability boundary. Table 4 shows gains of 3.56/2.34 on Qwen3.5-4B and 3.15/2.83 on InternVL3-2B. Three Qwen3.5-2B runs yield a compositional average of $8 6 . 7 3 \pm 0 . 2 7$ but ViVerBench, MMMU, and MathVista regress while MMBench improves and POPE remains stable (Appendix C.3). Multi-edit curricula, unseen perturbation types, and caption filtering are studied in Appendices D and E. Figure 4 illustrates the learned verifier behavior. Additional failure analysis and task extensions are provided in Appendix G.

Limitations. PROVE uses four types of localized caption perturbations, which align with the evaluated compositional axes and may saturate as the Repairer improves. Although independent audits support improved visual correctness, an evolving shared Verifier can still reinforce its own perceptual mistakes. Training depends on faithful starting captions from one corpus and requires four role-conditioned calls per trajectory, increasing compute relative to static supervision. Broader capability changes are benchmark-dependent, including regressions on ViVerBench, MMMU, and MathVista. Task-specific self-play for VQA, grounding, and referring expressions remains untested. Imperfect verifiers can also propagate errors when used for automated evaluation or filtering; these uses require independent validation.

## 5 Conclusion

We introduced PROVE, a self-play framework for training vision-language verifiers through a closed perturb–verify–repair–verify loop. Instead of relying on stronger external judges, fixed hard-negative datasets, or static preference annotations, PROVE converts ordinary clean image-caption pairs into adaptive verification episodes. A single shared MLLM acts as a Mutator, Verifier, and Repairer under different role prompts: the Mutator creates localized image-text mismatches, the Verifier diagnoses them, and the Repairer attempts to recover faithful captions from visual evidence and verifier feedback. This design turns verification into an operational self-improvement problem, where the model learns not only to judge image-text faithfulness but also to expose and correct its own fine-grained failure modes. The resulting training signal is grounded, adaptive, and verifier-guided. Typed perturbations over objects, attributes, counts, and relations provide controllable supervision for compositional understanding, while the two verification rounds produce rewards that update the same backbone used at test time. Experiments on Qwen3.5-2B, Qwen3.5-4B, and InternVL3-2B backbones show that PROVE improves fine-grained compositional understanding across multiple benchmarks. On the reference backbone, it also strengthens visual-detail sensitivity, with limited, benchmark-dependent changes on broader multimodal tasks. These results suggest that self-play over perturbation, verification, and repair offers a practical alternative to static judge distillation for building stronger vision-language verifiers.

## Acknowledgments

This work was supported by the European Union’s Horizon Europe research and innovation programme under grant agreement number 101214398 (ELLIOT).

## References

[1] Lorenzo Bianchi, Fabio Carrara, Nicola Messina, Claudio Gennaro, and Fabrizio Falchi. The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. arXiv:2311.17518.

[2] Kyle Buettner and Adriana Kovashka. Investigating the role of attribute context in visionlanguage models for object recognition and detection. In IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 2024. arXiv:2303.10093.

[3] Jirayu Burapacheep, Ishan Gaur, Agam Bhatia, and Tristan Thrush. ColorSwap: A color and word order dataset for multimodal evaluation. In Findings of the Association for Computational Linguistics: ACL 2024, 2024. arXiv:2402.04492.

[4] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. ShareGPT-4o-Image: Aligning multimodal models with GPT-4o-level image generation, 2025. URL https://arxiv.org/abs/2506.18095.

[5] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to GPT-4V? closing the gap to commercial multimodal models with open-source suites, 2024. URL https://arxiv.org/abs/2404.16821. Technical report for the InternVL2 series.

[6] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.06500.

[7] Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-Bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, Shimon Ullman, and Leonid Karlinsky. Dense and aligned captions (DAC) promote compositional reasoning in VL models. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2305.19595.

[8] Yingjun Du, Wenfang Sun, and Cees G Snoek. Ipo: Interpretable prompt optimization for visionlanguage models. Advances in Neural Information Processing Systems, 37:126725–126766, 2024.

[9] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusion-Bench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, 2024. arXiv:2310.14566.

[10] Yicheng He, Chengsong Huang, Zongxia Li, Jiaxin Huang, and Yonghui Yang. VisPlay: Selfevolving vision-language models from images. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026. arXiv:2511.15661.

[11] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. SugarCrepe: Fixing hackable benchmarks for vision-language compositionality. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023. arXiv:2306.14610.

[12] Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-Zero: Self-evolving reasoning LLM from zero data. In International Conference on Learning Representations (ICLR), 2026. arXiv:2508.05004.

[13] Geoffrey Irving, Paul Christiano, and Dario Amodei. AI safety via debate, 2018. URL https://arxiv.org/abs/1805.00899.

[14] Zexi Jia, Chuanwei Huang, Hongyan Fei, Yeshuang Zhu, Zhiqiang Yuan, Ying Deng, Jiapei Zhang, Jinchao Zhang, and Jie Zhou. A visual leap in CLIP compositionality reasoning through generation of counterfactual sets. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 23498–23507, 2025. arXiv:2507.04699.

[15] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Min Joon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision (ECCV), pages 235–251, 2016.

[16] Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. RLAIF vs. RLHF: Scaling reinforcement learning from human feedback with AI feedback. In International Conference on Machine Learning (ICML), 2024. arXiv:2309.00267.

[17] Haoxin Li and Boyang Li. Enhancing vision-language compositional understanding with multimodal synthetic data. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24849–24861, 2025. arXiv:2503.01167.

[18] Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, Lingpeng Kong, and Qi Liu. VLFeedback: A large-scale AI feedback dataset for large vision-language models alignment. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2024. arXiv:2410.09421.

[19] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023. arXiv:2305.10355.

[20] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations (ICLR), 2024. arXiv:2305.20050.

[21] Inna Wanyin Lin, Yushi Hu, Shuyue Stella Li, Scott Geng, Pang Wei Koh, Luke Zettlemoyer, Tim Althoff, and Marjan Ghazvininejad. Self-improving VLM judges without human annotations, 2025. URL https://arxiv.org/abs/2512.05145.

[22] Zhiqiu Lin, Xinyue Chen, Deepak Pathak, Pengchuan Zhang, and Deva Ramanan. Revisiting the role of language priors in vision-language models. In International Conference on Machine Learning (ICML), 2024. arXiv:2306.01879; introduces the VisualGPTScore for visio-linguistic reasoning.

[23] Bo Liu, Chuanyang Jin, Seungone Kim, Weizhe Yuan, Wenting Zhao, Ilia Kulikov, Xian Li, Sainbayar Sukhbaatar, Jack Lanchantin, and Jason Weston. SPICE: Self-play in corpus environments improves reasoning, 2025. URL https://arxiv.org/abs/2510.24684.

[24] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 26296–26306, 2024. arXiv:2310.03744.

[25] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision (ECCV), pages 216–233, 2024. arXiv:2307.06281.

[26] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: On the hidden mystery of OCR in large multimodal models. Science China Information Sciences, 67(12), 2024. doi: 10.1007/s11432-024-4235-6.

[27] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective. In Conference on Language Modeling (COLM), 2025. Proposes Dr.GRPO, which removes the standard-deviation normalization in GRPO; arXiv:2503.20783.

[28] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024. arXiv:2310.02255.

[29] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2203.02155.

[30] Fiorenzo Parascandolo, Nicholas Moratelli, Enver Sangineto, Lorenzo Baraldi, and Rita Cucchiara. Causal graphical models for vision-language compositional understanding. In International Conference on Learning Representations (ICLR), 2025. arXiv:2412.09353.

[31] Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents. https: //qwen.ai/blog?id=qwen3.5, February 2026.

[32] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021. arXiv:2103.00020.

[33] Ugur Sahin, Hang Li, Qadeer Khan, Daniel Cremers, and Volker Tresp. Enhancing multimodal compositional reasoning of visual language models with generative negative mining. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 5551–5561, 2024. arXiv:2311.03964.

[34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300. Introduces Group Relative Policy Optimization (GRPO).

[35] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems (EuroSys), pages 1279–1297, 2025. doi: 10.1145/3689031.3696075.

[36] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play. Science, 362(6419):1140–1144, 2018.

[37] Wenfang Sun, Hao Chen, Yingjun Du, Yefeng Zheng, and Cees G Snoek. Regionreasoner: Region-grounded multi-round visual reasoning. In International Conference on Learning Representations, volume 2026, pages 80583–80603, 2026.

[38] Wenfang Sun, Yingjun Du, and Cees GM Snoek. Less from more: Reinforcing sparse video reasoning from dense references. arXiv preprint arXiv:2610.10893, 2026.

[39] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. Aligning large multimodal models with factually augmented RLHF. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, 2024. arXiv:2309.14525.

[40] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5238–5248, 2022.

[41] Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal LLMs. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9568–9578, 2024. arXiv:2401.06209.

[42] Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, and Lucas Beyer. Image captioners are scalable vision learners too. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2306.07915.

[43] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9556–9567, 2024. arXiv:2311.16502.

[44] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In International Conference on Learning Representations (ICLR), 2023. arXiv:2210.01936.

[45] Yan Zeng, Xinsong Zhang, and Hang Li. Multi-grained vision language pre-training: Aligning texts with visual concepts. In International Conference on Machine Learning (ICML), pages 25994–26009, 2022. arXiv:2111.08276.

[46] Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, Yujiu Yang, and Guang Shi. Generative universal verifier as multimodal metareasoner. In International Conference on Learning Representations (ICLR), 2026. Introduces OmniVerifier-7B and the ViVerBench benchmark; arXiv:2510.13804.

[47] Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2505.03335.

[48] Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. VL-CheckList: Evaluating pre-trained vision-language models with objects, attributes and relations, 2022. URL https://arxiv.org/abs/2207.00221.

[49] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023. arXiv:2306.05685.

[50] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In International Conference on Learning Representations (ICLR), 2024. arXiv:2304.10592.

## A Additional Experimental Details

## A.1 Training Hyperparameters

Table 5 summarizes the main hyperparameters used for the Qwen3.5-2B reference run. The same shared backbone is used for the Mutator, Verifier, and Repairer roles, with role identity specified by prompt templates.

Table 5: Training hyperparameters for the Qwen3.5-2B reference run.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone Training data Validation data</td><td>Qwen3.5-2B 48k image-caption pairs 5k image-caption pairs</td></tr><tr><td>max prompt length / response length rollout batch size</td><td>4096 / 1024 48</td></tr><tr><td>rollout size n</td><td>3</td></tr><tr><td>sampling temperature / top-p / top-k</td><td>0.7 / 0.8 / 20</td></tr><tr><td>presence penalty actor learning rate</td><td>1.5</td></tr><tr><td></td><td> $5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>global batch size</td><td></td></tr><tr><td></td><td>48</td></tr><tr><td>PPO clip range</td><td>[0.20, 0.22]</td></tr><tr><td>KL coefficient β</td><td>0.005</td></tr><tr><td>invalid-action penalty</td><td>-0.3</td></tr><tr><td>precision</td><td></td></tr><tr><td></td><td>BF16</td></tr><tr><td>hardware</td><td>8 NVIDIA A100 GPUs</td></tr><tr><td>reported checkpoint</td><td>best validation checkpoint</td></tr></table>

## A.2 Additional Benchmark Results

Table 6 reports checkpoint-level results on additional benchmarks [15, 26, 28, 43, 46]. The checkpoints are denoted as PROVE @450, PROVE @600, and PROVE @750 according to the training step. The main paper reports the best validation checkpoint, while this table provides the full trajectory for completeness.

Table 6: Extended checkpoint results for Qwen3.5-2B. PROVE @450, PROVE @600, and PROVE @750 denote checkpoints at steps 450, 600, and 750, respectively.
<table><tr><td>Benchmark</td><td>Base</td><td>PROVE @450</td><td>PROVE @600</td><td>PROVE @750</td></tr><tr><td>Winoground (2-AFC)</td><td>69.50</td><td>74.50</td><td>74.75</td><td>75.62</td></tr><tr><td>ARO macro</td><td>73.57</td><td>78.01</td><td>78.27</td><td>78.66</td></tr><tr><td>SugarCrepe micro</td><td>93.76</td><td>94.24</td><td>94.28</td><td>94.17</td></tr><tr><td>VL-CheckList 2-axis avg</td><td>79.80</td><td>83.58</td><td>83.87</td><td>84.20</td></tr><tr><td>POPE F1</td><td>89.57</td><td>89.52</td><td>89.58</td><td>89.66</td></tr><tr><td>MMVP Pair Acc.</td><td>56.00</td><td>59.33</td><td>57.33</td><td>60.00</td></tr><tr><td>ViVerBench (rule)</td><td>55.16</td><td>52.81</td><td>52.79</td><td>52.98</td></tr><tr><td>MMBench-EN dev</td><td>71.31</td><td>72.16</td><td>71.65</td><td>72.34</td></tr><tr><td>OCRBench</td><td>84.50</td><td></td><td>84.90</td><td>84.90</td></tr><tr><td>MMMU val</td><td>40.78</td><td></td><td>40.11</td><td>40.44</td></tr><tr><td>MathVista testmini</td><td>73.80</td><td>74.50</td><td>73.60</td><td>72.80</td></tr><tr><td>AI2D</td><td>75.26</td><td></td><td>75.71</td><td>75.91</td></tr></table>

Table 7 reports the VL-CheckList axis-level results. The gains are consistent across attribute, object, and relation axes, showing that PROVE improves localized image-text verification rather than only a single benchmark score.

Table 7: VL-CheckList axis-level results on Qwen3.5-2B.
<table><tr><td>Metric</td><td>Base</td><td>PROVE @450</td><td>PROVE @600</td><td>PROVE @750</td></tr><tr><td>Attribute axis</td><td>76.75</td><td>80.72</td><td>81.13</td><td>81.46</td></tr><tr><td>Object axis</td><td>91.20</td><td></td><td></td><td>94.14</td></tr><tr><td>Relation axis</td><td>82.85</td><td>86.45</td><td>86.60</td><td>86.95</td></tr><tr><td>3-axis average</td><td>83.60</td><td>一</td><td></td><td>87.52</td></tr></table>

## A.3 Reward Design Discussion

The verifier-guided rewards in PROVE are designed to couple two complementary goals: exposing verifier weaknesses and recovering faithful captions. For the Mutator, the reward contains an exposure term, a hardness term, and an edit-cost penalty. The exposure term encourages perturbations that the current verifier fails to reject, while the hardness term encourages perturbations that remain challenging after repair. The cost term discourages overly wide edits. Actions rejected by the deterministic editor receive a fixed penalty instead of $C _ { M }$ (Appendix F).

For the Repairer, the reward contains an acceptance term, a judgment-flip term, and a repair-cost penalty. The acceptance term encourages repaired captions that the verifier judges faithful, while the judgment-flip term rewards a change in the verifier decision, $\mathcal { H } [ y _ { 1 } \ne y _ { 2 } ]$ ; the separate acceptance term favors a faithful accepted repair. The repair-cost penalty discourages unnecessary rewriting and encourages localized correction.

Table 8 summarizes the role of each reward component.

Table 8: Verifier-guided reward components in PROVE.
<table><tr><td>Role</td><td>Component</td><td>Purpose</td></tr><tr><td>Mutator</td><td> $R _ { \mathrm { e x p o s e } } ( y _ { 1 } )$ </td><td>Expose verifier blind spots</td></tr><tr><td>Mutator</td><td> $R _ { \mathrm { h a r d } } ( y _ { 2 } )$ </td><td>Prefer perturbations that remain challenging after repair</td></tr><tr><td>Mutator</td><td> $C _ { M } ( { a } _ { M } )$ </td><td>Penalize overly wide edits</td></tr><tr><td>Mutator</td><td>invalid-action penalty</td><td>Fixed penalty when the editor rejects the action</td></tr><tr><td>Repairer</td><td> $R _ { \mathrm { a c c e p t } } ( y _ { 2 } )$ </td><td>Reward repaired captions accepted by the verifier</td></tr><tr><td>Repairer</td><td> $R _ { \mathrm { f l i p } } ( y _ { 1 } , y _ { 2 } )$ </td><td>Reward a change in the verifier judgment</td></tr><tr><td>Repairer</td><td> $C _ { R } ( p ^ { \prime } , \tilde { p } )$ </td><td>Penalize unnecessary rewriting</td></tr></table>

## A.4 Compute Cost

The Qwen3.5-2B reference run is trained on a single node with 8 NVIDIA A100 GPUs using BF16 mixed precision. Each self-play episode executes the perturb–verify–repair–verify trajectory with one shared backbone under different role prompts. The main run uses rollout size $n = 3$ and global batch size 48; the reported trajectory includes checkpoints through step 750. The reference run takes 11 hours on 8 A100 GPUs, totaling 88 GPU-hours. We will release the training and evaluation code, including launch scripts and prompt templates, to facilitate reproducibility.

We profile all complete pipelines on the same hardware and 48k source pairs (Table 9). For PROVE, supervision is generated online and is included in target-training time. Its four role-conditioned calls per trajectory produce more generated tokens than either static negatives or offline external-judge distillation. Thus, the benefit is adaptive supervision without a separate teacher or fixed negative set, rather than universally lower raw compute.

Table 9: Complete pipeline compute costs. Generation and training costs are GPU-hours. Online generation for PROVE is included in its 88.0 GPU-hours.
<table><tr><td>Method</td><td>Supervision generation</td><td>Target training</td><td>Total GPU-hours</td><td>Generated tokens/source</td></tr><tr><td>Fixed typed negatives</td><td>4.8</td><td>43.2</td><td>48.0</td><td>46</td></tr><tr><td>External-judge distillation</td><td>27.6</td><td>45.3</td><td>72.9</td><td>214</td></tr><tr><td>PROVE</td><td>Online</td><td>88.0</td><td>88.0</td><td>1,126</td></tr></table>

## A.5 Illustrative Examples of the PROVE Pipeline

To complement the method description in Section 3, we provide end-to-end examples for the four supported Mutator perturbation types: count, attribute, relation, and object. Each example follows the perturb–verify–repair–verify trajectory:

$$
\left( x , p \right) \xrightarrow { \mathrm { M u t a t o r } } p ^ { \prime } \xrightarrow [ ] { \mathrm { V e r i f i e r } } \left( y _ { 1 } , r _ { 1 } \right) \xrightarrow [ ] { \mathrm { R e p a i r e r } } \tilde { p } \xrightarrow [ ] { \mathrm { V e r i f i e r } } \left( y _ { 2 } , r _ { 2 } \right) .
$$

The Mutator emits a localized edit of the form (edit\_type, target\_text, replacement), which is applied to the clean caption $p$ to obtain the perturbed caption $p ^ { \prime }$ . The Verifier first judges whether $p ^ { \prime }$ matches the image and returns a binary decision $y _ { 1 }$ with textual feedback $r _ { 1 }$ . Conditioned on the image, the perturbed caption, and this feedback, the Repairer directly generates a repaired caption ${ \tilde { p } } .$ The Verifier is then invoked again to obtain $( y _ { 2 } , r _ { 2 } )$ . These examples illustrate how PROVE converts clean image-caption pairs into grounded hard negatives and repair targets without requiring a stronger external judge.

## A.5.1 Count

Illustrative PROVE Pipeline – Count Action  
![](images/25df7e04bf024cd012166a35b9036429b7df13ebe6ff7923a9df03a854bec554.jpg)

## Original prompt p (label = 1):

“Afront view shot, utilizing strong vertical linesfor composition, capturesfour divers before a central altar laden with mysterious offerings, their variedforms: thefirst, a technical specialist with a backplate and wing, twin steel tanks, a sealed drysuit, separate thick neoprene hood, and powerful blade fins; the second in afull wetsuit employing a sidemount BCD with a single tank, theirface obscured by a full-face mask with integrated regulator, and distinguished by camouflage patternedfins; a third clearly recreational, with a jacket-style BCD, single aluminum tank, low-volume dual-lens mask, and standard paddle fins; and the fourth minimally equipped in a shorty wetsuit and full-foot fins—dramatically silhouetted by a strong, warm backlight yet softly defined by diffused ambient light, all rendered in an often high-contrast black and white artistic style.”

Mutator action: {"edit\_type": "count", "target\_text": "four divers", "replacement": "five divers"}

Perturbed prompt $p ^ { \prime }$ (label = 0):

“Afront view shot . . . capturesfive divers before a central altar laden with mysterious offerings, their variedforms: thefirst . . . and thefourth minimally equipped in a shorty wetsuit andfull-footfins—. . . Verifier (1st pass) on $( x , p ^ { \prime } ) \colon$ <think> The user wants me to verify whether the description matches the image. Analyze the image: four divers stand in front of a central stone altar; the altar holds a skull, several candles, and small jugs; a strong vertical backlight produces silhouettes against a high-contrast black-and-white scene. . . Analyze the description: $^ { 6 6 } \cdot \cdot \cdot$ captures five divers . . . ” –- Mismatch: the image clearly shows four divers, not five. The altar with offerings, the warm backlight, and the B&W styling all align with the rest of the prompt; only the count claim is violated. . . Conclusion: the description does not accurately match the image. </think>

{ "answer": false, "explanation": "The image contains four divers around the altar, not five." }

⇒ y<sub>1</sub> = false, r<sub>1</sub> = “The image contains four divers around the altar, not five.”

## Repairer output p˜: Repairer output p:

“Afront view shot, utilizing strong vertical linesfor composition, capturesfour divers before a central altar laden with mysterious offerings . . . ” (recovers the count slot in p)   
Verifier (2nd pass) on (x, p˜): <think> Re-analyze the image: four figures stand in a row before the central altar with a skull, candles, and small jugs; the vertical god-ray and high-contrast B&W styling are unchanged from the previous pass. . . Re-analyze the description: “. . . captures four divers . . . ” –- the count is now correct, and the altar, the warm backlight, and the B&W rendering all line up with the description. . . Conclusion: modulo a few gear-level paraphrases the description is consistent with the image. </think>   
{ "answer": true, "explanation": "Four divers are visible around the altar, matching the description." }   
⇒ y<sub>2</sub> = true. Alignment score s ≈ 0.995 (p˜ near-identical to p, count slot fully restored).

## A.5.2 Attribute

Illustrative PROVE Pipeline – Attribute Action  
![](images/3cd63d090d85a221678812accd1de7ae94144f3de1f58a813125e18c9b4e05c9.jpg)

## Original prompt p (label = 1):

“A bald chibi character stands with hands on his hips, wearing an orange martial arts uniform with a yellow circular logo on the chest, a black belt around the waist, and blue wristbands on both arms.”

Mutator action: {"edit\_type": "attribute", "target\_text": "an orange martial arts uniform", "replacement": "a green martial arts uniform"}   
Perturbed prompt p<sup>′</sup> (label = 0):   
“A bald chibi character stands with hands on his hips, wearing a green martial arts uniform with a yellow circular logo on the chest, a black belt around the waist, and blue wristbands on both arms.” Verifier (1st pass) on (x, p<sup>′</sup>): <think> The user wants me to verify whether the description matches the image. Analyze the image: a bald chibi character stands with hands on his hips, wearing an unmistakably orange martial-arts gi; the chest carries a yellow circular Dragon-Ball-style logo, a black belt is tied at the waist, and blue wristbands are visible on both wrists. . . Analyze the description: “. . . wearing a green martial arts uniform . . . yellow circular logo . . . black belt . . . blue wristbands . . . ” –- Mismatch on the uniform colour (green vs. orange ); the logo, belt, and wristbands all match. . . Conclusion: the description has a single colour error on the uniform. </think>   
{ "answer": false, "explanation": "The uniform is orange, not green." }   
⇒ y<sub>1</sub> = false, r<sub>1</sub> = “The uniform is orange, not green.”

Repairer output p˜:   
“A bald chibi character stands with hands on his hips, wearing an orange martial arts uniform with a   
yellow circular logo on the chest, a black belt around the waist, and blue wristbands on both arms.”   
(identical to p, attribute slot recovered)   
Verifier (2nd pass) on (x, p˜): <think> Re-analyze the image: bald chibi character, hands on hips,   
orange gi, yellow circular logo on the chest, black belt at the waist, blue wristbands on both   
wrists. . . Re-analyze the description: “. . . orange martial arts uniform . . . yellow circular logo   
. . . black belt . . . blue wristbands . . . ” –- every attribute slot now agrees with the image, and   
no further mismatch remains. . . Conclusion: the description is an accurate match for the image.   
</think>   
{ "answer": true, "explanation": "All visual attributes (orange gi, yellow logo, black belt,   
blue wristbands) match the description." }   
⇒ y<sub>2</sub> = true. Alignment score s ≈ 0.99 (single attribute slot restored).

## A.5.3 Relation

Illustrative PROVE Pipeline – Relation Action  
![](images/0ca621cf0a9ab0b978c6217bfd4cb1e9b341085b0a89814db3962c2ef50ced63.jpg)

## Original prompt p (label = 1):

“A person wearing a white GAP hoodie is standing indoors with arms raised, holding a coiled white cable in both hands, near an open window with curtains.”

Mutator action: {"edit\_type": "relation", "target\_text": "standing indoors", "replacement": "standing outside"}

## Perturbed prompt p<sup>′</sup> (label = 0):

“A person wearing a white GAP hoodie is standing outside with arms raised, holding a coiled white cable in both hands, near an open window with curtains.”

Verifier (1st pass) on (x, p<sup>′</sup>): <think> The user wants me to verify whether the description matches the image. Analyze the image: a young man with dark hair and a beard wears a white GAP hoodie; his arms are raised above his head and his hands hold a coiled white cable; behind him, heavy beige curtains hang on the left and a white-framed window is visible on the right, with soft diffuse interior lighting. . . Analyze the description: “. . . standing outside . . . ” –- Mismatch: the curtains, the window frame, and the indoor lighting all indicate an interior space, not an outdoor one. The hoodie, raised arms, and coiled cable do match. . . Conclusion: the spatial relation in the prompt is inverted with respect to the image. </think>   
{ "answer": false, "explanation": "The person is standing indoors, not outside." }   
⇒ y<sub>1</sub> = false, r<sub>1</sub> = “The person is standing indoors, not outside.”

## Repairer output p˜:

“A person wearing a white GAP hoodie is indoors with arms raised, holding a coiled white cable in both hands, near an open window with curtains.”

Verifier (2nd pass) on (x, p˜): <think> Re-analyze the image: indoor scene with beige curtains on the left and a white-framed window on the right; a man in a white GAP hoodie raises his arms above his head and holds a coiled white cable in both hands. . . Re-analyze the description: “. . .

indoors with arms raised, holding a coiled white cable . . . near an open window with curtains” –-   
every clause aligns with the image: the indoor relation, the raised arms, the coiled cable, and   
the window with curtains. . . Conclusion: the description accurately matches the image. </think>   
{ "answer": true, "explanation": "The person is indoors wearing a white GAP hoodie with arms   
raised holding a cable, near an open window with curtains." }   
⇒ y<sub>2</sub> = true. Alignment score s = 0.958 (relation slot recovered; “standing” dropped, otherwise   
close to p).

## A.5.4 Object

![](images/9b64ea5e0d59732da35be844bd531d10ae718f745a1a29611118e7be27faf7a4.jpg)  
Takeaways. Across the four edit types, the same minimal-edit principle holds: the Mutator flips exactly one semantic slot (count, attribute, relation, or object) via a structured edit, the first

Verifier pass localises the mismatch via a short rationale, and the Repairer uses that rationale to directly generate a faithful caption $\tilde { p }$ that, modulo at most a few-token paraphrase (e.g. “standing indoors” → “indoors”), matches the original $p .$ The second Verifier pass closes the loop and produces the success signal $y _ { 2 }$ that drives the Repairer reward $r _ { R }$ (Eq. 12). Because perturbation is a discrete typed action while repair is free-form caption generation, the two roles are not symmetric: the Mutator is constrained to expose verifier blind spots along a known axis, whereas the Repairer must recover the full caption from visual evidence and verifier feedback, which is what makes repair an operational form of verification.

## B Controlled Supervision and Loop Ablations

Matched supervision settings. All comparisons use the same Qwen3.5-2B initialization, 48k clean image-caption pairs, target-model optimization budget, and evaluation protocol. Fixed typed negatives are generated once and reused throughout training. External-judge distillation uses judgments and textual feedback from a frozen larger VLM. Verifier self-training uses self-generated labels without the Repairer or second verification. This baseline isolates self-generated verifier supervision under a matched image-text protocol; it is not a reproduction of reasoning-oriented self-play methods with different tasks and supervision. Table 3 reports the matched supervision results. The same optimization budget does not imply equal GPU-hours, since the methods execute different generation pipelines (Table 9).

## B.1 Component and Reward Ablations

Table 10 isolates the role of static versus adaptive perturbations, the repair loop, verifier feedback, the optimizer, and reward components. Removing the Repairer and second verification produces the largest drop, 2.61/1.72 points on the compositional/broader averages. Removing feedback costs 1.68/1.19 points, indicating that repair benefits from the verifier’s diagnosis. Both role rewards contribute: removing the Mutator or Repairer reward costs 1.82/1.27 or 1.31/0.94 points, respectively. Removing the edit and repair costs reduces the averages by 0.77/0.60 points. Replacing DrGRPO with GRPO produces a smaller 0.43/0.32 drop; the main gain therefore depends on the closed-loop supervision rather than the optimizer alone.

Table 10: Complete component, optimizer, and reward ablations. All variants use the same source data, backbone initialization, target-model optimization budget, and evaluation protocol; the GRPO row changes only the optimization variant.
<table><tr><td>Training variant</td><td>Comp. Avg.</td><td>Broader Avg.</td></tr><tr><td>Qwen3.5-2B</td><td>83.22</td><td>58.77</td></tr><tr><td>Fixed typed negatives</td><td>85.02</td><td>59.68</td></tr><tr><td>No Repairer or second verification</td><td>84.24</td><td>59.31</td></tr><tr><td>Repairer without verifier feedback</td><td>85.17</td><td>59.84</td></tr><tr><td>GRPO instead of DrGRPO</td><td>86.42</td><td>60.71</td></tr><tr><td>Without Mutator reward</td><td>85.03</td><td>59.76</td></tr><tr><td>Without Repairer reward</td><td>85.54</td><td>60.09</td></tr><tr><td>Without edit/repair costs</td><td>86.08</td><td>60.43</td></tr><tr><td>PROVE</td><td>86.85</td><td>61.03</td></tr></table>

## B.2 Role Configuration and Verifier Updating

Table 11 compares shared role prompts with role-specific adapters and frozen Verifier variants. Freezing the internal Verifier lowers the compositional/broader averages by 1.73/1.22 points, supporting verifier co-evolution. Role-specific adapters remain competitive but are 0.44/0.25 points below full parameter sharing. A frozen external Verifier is stronger than the frozen internal Verifier, but remains 0.78/0.49 points below the shared loop. This frozen external Verifier operates online inside the self-play loop, whereas external-judge distillation uses offline teacher judgments as target supervision. These are distinct comparisons.

Table 11: Role configuration and verifier-update ablations.
<table><tr><td>Role configuration</td><td>Comp. Avg.</td><td>Broader Avg.</td></tr><tr><td>Shared backbone, role prompts</td><td>86.85</td><td>61.03</td></tr><tr><td>Shared backbone, role-specific adapters</td><td>86.41</td><td>60.78</td></tr><tr><td>Frozen internal Verifier</td><td>85.12</td><td>59.81</td></tr><tr><td>Frozen external Verifier</td><td>86.07</td><td>60.54</td></tr></table>

## C Independent Verification and Repeated-Run Analysis

## C.1 External Trajectory Audit

We audit a held-out trajectory set using labels produced outside the self-play loop. These labels are never used for training, reward computation, mutation generation, validation, or checkpoint selection. The initial Qwen3.5-2B model and PROVE use the same role prompts and deterministic editor. Table 12 reports verifier accuracy on a fixed set and the quality of the generated trajectories.

A valid localized perturbation introduces a real image-caption mismatch confined to the intended edit. A correct verifier judgment identifies whether the caption is faithful to the image. A faithful minimal repair corrects the mismatch without introducing unsupported changes. A fully valid trajectory requires all four stages to be correct: a real localized mismatch, a correct diagnosis, a faithful minimal repair, and a correct final judgment. The fully valid rate increases by 23.4 points, while fixed-set verifier accuracy increases by 7.8 points. These external measurements support improved visual correctness, while the shared-model setting still leaves possible self-verification bias (Section 4.3).

Table 12: Complete external audit. All entries are percentages. Fixed-set verifier accuracy and trajectory-quality rates are separate audit measurements.
<table><tr><td>External audit</td><td>Base</td><td>PROVE</td></tr><tr><td>Fixed-set verifier accuracy</td><td>75.8</td><td>83.6</td></tr><tr><td>Valid localized perturbations</td><td>82.4</td><td>92.1</td></tr><tr><td>Correct verifier judgments</td><td>71.5</td><td>84.7</td></tr><tr><td>Faithful minimal repairs</td><td>55.2</td><td>73.4</td></tr><tr><td>Fully valid trajectories</td><td>40.8</td><td>64.2</td></tr></table>

## C.2 External Benchmark Trajectory

We evaluate checkpoints on fixed, externally labeled benchmarks that are excluded from training, self-play rewards, mutation generation, validation, and checkpoint selection. Table 13 separates the internal self-play pass rate from externally measured benchmark performance. The compositional average increases from 83.22 to 86.85, while the broader average increases from 58.77 to 61.03. The broader trajectory is not monotonic: its step-600 value is below step 450, illustrating why internal agreement is not a substitute for independent evaluation. Across the six reported checkpoints, Pearson correlations between pass rate and the compositional/broader averages are 0.9254/0.9022. These descriptive correlations summarize co-movement during training; they do not establish a causal relationship or a significance test. Figure 5 plots the six reported measurements without adding intermediate observations.

Table 13: Internal self-play pass rate and external benchmark averages across training. The pass rate is in percent; benchmark averages use the main-paper definitions.
<table><tr><td>Checkpoint</td><td>Self-play pass rate</td><td>Comp. Avg.</td><td>Broader Avg.</td></tr><tr><td>Base</td><td>36.0</td><td>83.22</td><td>58.77</td></tr><tr><td>Step 150</td><td>39.3</td><td>84.08</td><td>59.18</td></tr><tr><td>Step 300</td><td>43.2</td><td>85.14</td><td>59.73</td></tr><tr><td>Step 450</td><td>46.3</td><td>85.97</td><td>60.41</td></tr><tr><td>Step 600</td><td>56.6</td><td>86.48</td><td>60.16</td></tr><tr><td>Step 750</td><td>63.3</td><td>86.85</td><td>61.03</td></tr></table>

![](images/09c7762f6c11f9b6aa33a09e367d32ea72333665c0be3808577db144b6dc052c.jpg)

![](images/492b0b936bdb31e075daa5394bee1c0fb78b5d94b744af01cc91681e5633cbb2.jpg)  
Figure 5: External benchmark performance and internal self-play pass rate. Each marker is one reported checkpoint; connecting lines guide the eye. Gains in the left panel are measured against the base checkpoint. The right panel reports internal pass rate separately, preserving the distinction between independent accuracy and role agreement.

## C.3 Repeated-Run Variability and Broader Capability

We repeat Qwen3.5-2B training with three independent seeds. Table 14 reports the mean and standard deviation across the three trained models. These statistics measure training-run variability, not example-level uncertainty, confidence intervals, or statistical significance. The reference-run results in Tables 1 and 2 and the checkpoint table in Appendix A.2 remain single-run measurements.

The core compositional average improves by 3.51 points across seeds. Broader changes are limited and task-dependent: ViVerBench regresses by 2.04 points, MMMU and MathVista decrease by 0.39 and 0.86, MMBench improves by 0.95, and POPE remains nearly unchanged. Thus, PROVE substantially improves compositional verification without implying uniform preservation or improvement of all broader capabilities.

Table 14: Results across three independent Qwen3.5-2B training runs. Change is the mean PROVE result minus the base score.
<table><tr><td>Benchmark</td><td>Base</td><td>PROVE (mean ± std)</td><td>Change</td></tr><tr><td>ViVerBench</td><td>55.16</td><td> $5 3 . 1 2 \pm 0 . 5 8$ </td><td>-2.04</td></tr><tr><td>MMMU</td><td>40.78</td><td> $4 0 . 3 9 \pm 0 . 4 7$ </td><td>-0.39</td></tr><tr><td>MathVista</td><td>73.80</td><td> $7 2 . 9 4 \pm 0 . 5 2$ </td><td>-0.86</td></tr><tr><td>MMBench</td><td>71.31</td><td> $7 2 . 2 6 \pm 0 . 3 1$ </td><td>+0.95</td></tr><tr><td>POPE</td><td>89.57</td><td> $8 9 . 6 4 \pm 0 . 0 9$ </td><td>+0.07</td></tr><tr><td>Core compositional average</td><td>83.22</td><td> $8 6 . 7 3 \pm 0 . 2 7$ </td><td>+3.51</td></tr></table>

## D Generality and Perturbation Curricula

## D.1 Backbone Scale and Family

We apply the same perturbation and reward formulation to Qwen3.5-4B and the scale-matched InternVL3-2B family. Table 4 in the main paper reports the full base and PROVE averages for all three backbones. The Qwen3.5-4B gains are 3.56/2.34 compositional/broader points, while InternVL3-2B gains 3.15/2.83. These results show that the gains extend beyond the 2B Qwen reference setting, although all runs still rely on one source training corpus.

## D.2 Leave-One-Type-Out Transfer

For each leave-one-type-out experiment, we remove the held-out type from the Mutator action space and resample the remaining types to preserve the same episode count and optimization budget. Table 15 reports performance on the held-out axis. The retained gain is $( s _ { \mathrm { L O O } } - s _ { \mathrm { b a s e } } ) / ( s _ { \mathrm { f u l l } } - s _ { \mathrm { b a s e } } )$

These axis-level scores follow the capability aggregation used in Figure 3(b), rather than the overall averages in the main tables. Leave-one-out training retains 48.2%–52.4% of the full-method gain across the three reported axes. This supports cross-type transfer while showing that direct exposure supplies additional axis-specific benefit.

Table 15: Leave-one-type-out transfer on the three reported axes. The held-out axis is absent from the training action space.
<table><tr><td>Held-out type</td><td>Base</td><td>Leave-one-out PROVE</td><td>Full PROVE</td><td>Retained gain</td></tr><tr><td>Object</td><td>86.1</td><td>87.2</td><td>88.2</td><td>52.4%</td></tr><tr><td>Attribute</td><td>78.1</td><td>81.1</td><td>84.1</td><td>50.0%</td></tr><tr><td>Relation</td><td>76.3</td><td>79.0</td><td>81.9</td><td>48.2%</td></tr></table>

## D.3 Single- and Multi-Edit Curricula

Table 16 compares one-edit, mixed one/two-edit, and two-edit curricula. The mixed curriculum raises the broader average by 0.15 points but lowers compositional performance and repair success. Using two edits throughout further reduces repair success to 48.6%. These results support single localized edits as the default because they simplify diagnosis, minimal repair, and credit assignment. A multi-edit extension may benefit from explicit partial-credit rewards, which are not evaluated in these comparisons.

Table 16: Perturbation curriculum comparison. Repair success is a percentage.
<table><tr><td>Curriculum</td><td>Comp. Avg.</td><td>Broader Avg.</td><td>Repair success</td></tr><tr><td>One edit</td><td>86.85</td><td>61.03</td><td>62.7</td></tr><tr><td>Mixture of one/two edits</td><td>86.06</td><td>61.18</td><td>57.4</td></tr><tr><td>Two edits</td><td>85.72</td><td>60.31</td><td>48.6</td></tr></table>

## E Positive-Caption Quality

We independently audit a random subset of the training pool, distinguishing fully faithful captions, minor unsupported details or ambiguity, and material image-caption mismatches. Table 17 reports the three fractions, which sum to 100%. Most captions are faithful, while 1.7% contain material mismatches. We then remove low-confidence positives and repeat training (Table 18). The two averages increase by 0.23/0.12 points, suggesting that the observed verification gain does not depend on retaining noisy positives. This does not remove the method’s dependence on sufficiently faithful reference captions.

Table 17: Independent audit of a random subset of the positive-caption pool.
<table><tr><td>Caption quality</td><td>Fraction</td></tr><tr><td>Fully faithful</td><td>90.1%</td></tr><tr><td>Minor unsupported detail or ambiguity</td><td>8.2%</td></tr><tr><td>Material image-caption mismatch</td><td>1.7%</td></tr></table>

## F Role Interfaces and Deterministic Editing

Role-conditioned interfaces. The Mutator receives the image and a faithful caption and returns a single structured action with fields edit\_type, target\_text, and replacement. The type is one of object, attribute, count, or relation. The deterministic editor applies the local replacement to construct the caption supplied to the first Verifier. The Verifier receives the image and candidate caption and produces a binary faithfulness decision with textual feedback; the illustrative outputs in Appendix A.5 use the fields answer and explanation. The Repairer receives the image, perturbed caption, first decision, and explanation, and directly generates a faithful caption with a minimal correction. The same Verifier interface is used for the second verification. The complete templates used in training are listed below.

Table 18: Training after filtering low-confidence positive captions.
<table><tr><td>Training pool</td><td>Comp. Avg.</td><td>Broader Avg.</td></tr><tr><td>Original 48k</td><td>86.85</td><td>61.03</td></tr><tr><td>Filtered positives</td><td>87.08</td><td>61.15</td></tr></table>

Prompt templates. <image> marks the image input. Braced fields are filled at run time: {p} is the clean caption p, {caption} is the candidate caption $( p ^ { \prime }$ in the first verification and $\tilde { p }$ in the second), {p’} is the perturbed caption, and {y1} and {r1} are the first Verifier decision and explanation.

Mutator prompt template   
<image>{p}   
You are a Mutator agent. Look at the image and the caption above, then propose a single minimal   
edit that turns the caption into a slightly incorrect description of the image.   
Output exactly one JSON object on a single line, with no extra text, no markdown fences, no   
comments, no thinking trace:   
{"edit\_type": "<object|attribute|count|relation>", "target\_text": "<exact substring from the   
caption>", "replacement": "<modified text, must differ from target\_text>"}   
Rules:   
- edit\_type is one of object, attribute, count, relation.   
- target\_text MUST be an exact substring that occurs once in the caption.   
- replacement MUST differ from target\_text.   
- Keep the edit minimal and locally semantic.   
- Output ONLY the JSON object, nothing else.

Verifier prompt template (both verifications)   
<image>{caption}   
You are a Verifier agent. Determine whether the caption faithfully describes the image.   
First write the reasoning inside <think> </think>. Then output exactly one JSON object:   
{"answer": <true|false>, "explanation": "<one short sentence>"}   
Rules:   
- answer is true only when every object, attribute, count, and relation in the caption matches the   
image.   
- answer is false when any object, attribute, count, or relation is wrong.   
- explanation names the mismatch, or states that the caption matches. Example: "The image   
contains four divers around the altar, not five."   
- Judge only what is visible. Do not rewrite the caption.

Repairer prompt template   
<image>{p’}   
You are a Repairer agent. The caption above may not faithfully describe the image. The verifier   
decision is y1={y1}, and its explanation is: {r1}   
Recover a caption that faithfully describes the image. Make the smallest correction needed and   
preserve the rest of the caption.   
Output only the repaired caption, with no JSON, no markdown fences, no comments, and no thinking   
trace.

Output parsing. The Mutator output is parsed as $a _ { M } = ( \tau , t , \delta )$ from edit\_type, target\_text, and replacement. The Verifier output follows the format of Appendix A.5: a reasoning trace inside <think> tags followed by the JSON object. answer=true gives $y = 1$ , answer=false gives $y = 0$ , and explanation is the textual feedback $r ;$ the two verifications yield $( y _ { 1 } , r _ { 1 } )$ and $( y _ { 2 } , r _ { 2 } )$ The Repairer output is the repaired caption $\tilde { p }$ itself rather than an edit action.

Deterministic editing constraints. A valid action has a supported edit type and nonempty target and replacement strings. The target must identify one unambiguous span in the caption, and the replacement must change it. Unsupported types, missing or ambiguous spans, unchanged replacements, and edits that cannot be applied deterministically are invalid. For a unique target span, the editor preserves the caption prefix and suffix and replaces only that span. Algorithm 1 specifies this local-edit interface. An invalid action receives the invalid-action penalty −0.3 in Table 5 as its Mutator reward, and $C _ { M }$ is not computed. Invalid edits do not establish a valid negative example.

Algorithm 1 Deterministic single-span editor specification   
Require: Caption $p ,$ structured action $a _ { M } = ( \tau , t , \delta )$   
1: if $a _ { M }$ cannot be parsed, $\tau \notin \mathcal { T }$ , or t, δ are missing or empty then   
2: return invalid action   
3: end if   
4: if $t = \delta$ or t does not identify exactly one span in $p$ then   
5: return invalid action   
6: end if   
7: Write $p = u$ t v using the unique target occurrence   
8: return $p ^ { \prime } = u \delta v$

Reward weights. Equations 10 and 12 instantiate Equations 9 and 11 with $R _ { \mathrm { e x p o s e } } ( y _ { 1 } ) = \lambda _ { 1 } \mathcal { V } [ y _ { 1 } =$ $1 ] , R _ { \mathrm { h a r d } } ( y _ { 2 } ) = \lambda _ { 2 } \bar { \mathcal { Y } } [ y _ { 2 } = 0 ]$ , and $\lambda _ { M } = \lambda _ { 3 }$ for the Mutator, and $R _ { \mathrm { a c c e p t } } ( y _ { 2 } ) = \beta _ { 1 } \mathcal { V } [ y _ { 2 } = 1 ]$ $R _ { \mathrm { H i p } } ( y _ { 1 } , y _ { 2 } ) = \beta _ { 2 } { \rlap / } k [ y _ { 1 } \neq y _ { 2 } ]$ , and $\lambda _ { R } = \beta _ { 3 }$ for the Repairer. Table 19 lists the weights used in all reported runs.

Table 19: Reward weights in Equations 10 and 12.
<table><tr><td>Weight</td><td>Value</td><td>Term</td></tr><tr><td> $\lambda _ { 1 }$ </td><td>1.0</td><td>Perturbation not rejected by the first verification</td></tr><tr><td> $\lambda _ { 2 }$ </td><td>0.5</td><td>Repaired caption still judged unfaithful</td></tr><tr><td> $\lambda _ { 3 }$ </td><td>0.3</td><td>Mutator edit cost  $C _ { M }$ </td></tr><tr><td> $\beta _ { 1 }$ </td><td>0.8</td><td>Repair accepted by the second verification</td></tr><tr><td> $\beta _ { 2 }$ </td><td>0.7</td><td>Judgment flip between the two verifications</td></tr><tr><td> $\beta _ { 3 }$ </td><td>0.3</td><td>Repairer rewriting cost  $C _ { R }$ </td></tr></table>

Cost terms. $C _ { M }$ penalizes overly wide edit spans, normalized by the character length of the clean caption:

$$
C _ { M } ( a _ { M } ) = 2 \cdot { \frac { \mathrm { l e n } ( t ) } { \mathrm { l e n } ( p ) } } ,\tag{16}
$$

with $C _ { M } = 0$ when t or p is empty. $C _ { R }$ penalizes unnecessary rewriting. Let $T ( \cdot )$ split a caption on whitespace and lowercase each token, and let lev be the token-level Levenshtein distance:

$$
C _ { R } ( p ^ { \prime } , \tilde { p } ) = \frac { \mathrm { l e v } \big ( T ( p ^ { \prime } ) , T ( \tilde { p } ) \big ) } { \operatorname* { m a x } \big ( | T ( p ^ { \prime } ) | , | T ( \tilde { p } ) | \big ) } .\tag{17}
$$

$C _ { R }$ is 0 for identical captions, approaches 1 for a complete rewrite, and is set to 1 when either caption has no tokens.

Reward and diagnostic interpretation. Equations 10 and 12 specify the exposure, hardness, acceptance, judgment-flip, and cost terms. The flip indicator $\mathcal { H } [ y _ { 1 } \ne y _ { 2 } ]$ rewards a decision change; it is not independently a guarantee of a visually correct repair. In particular, an evolving Verifier can accept an incorrect caption, motivating the external audit in Appendix C.1. Repair success, structural alignment, and internal pass rate are training diagnostics, whereas the external benchmark and audit labels measure correctness independently. For a group of $n = 3$ binary outcomes, the empirical success fraction can take values $0 , 1 / 3 , \dot { 2 } / 3 , 1$ . Its Bernoulli variance $\bar { y } ( 1 - \bar { y } )$ is maximized at both $1 / 3$ and $2 / 3 ;$ Figure 3 plots the lower finite-group reference. This differs from the population Bernoulli variance maximum at $1 / 2$ and is a diagnostic reference, not an additional measurement or an external accuracy target.

## G Axis-level Gains, Qualitative Behavior, and Task Extensions

## G.1 Axis-level Gains

Axis-level gains. Figure 3(b) summarizes where the gains of PROVE come from by aggregating benchmark splits according to the visual-semantic capability they test. The largest improvements appear on attribute and relation verification, which are directly aligned with the typed perturbations used by the Mutator. This pattern supports the intended training mechanism: localized perturbations create visually checkable mismatches, forcing the Verifier to diagnose fine-grained inconsistencies and the Repairer to recover faithful captions. The positive gain on visual-detail sensitivity further suggests that verifier-guided repair improves localized evidence grounding beyond the core compositional benchmarks. In contrast, object and color/order gains are smaller but still positive, partly because the base model already performs strongly on these axes.

## G.2 Qualitative Behavior and Failure Cases

Qualitative verifier behavior. Figure 4 provides representative examples of how the learned verifier changes after self-play training. The base model often accepts captions or answers that overlook a localized mismatch, such as an incorrect relation, missing object evidence, wrong attribute, or counting error. After PROVE training, the verifier produces judgments that better track the relevant visual evidence and reject mismatches that the base model misses. These examples indicate that PROVE does not merely improve aggregate scores; it changes the verifier’s behavior in the intended direction, making it more sensitive to the same visual-semantic factors targeted by the perturb–verify– repair loop.

Failure cases. Our error analysis identifies four recurring settings: small or occluded objects that are misperceived; ambiguous relations in crowded scenes or unusual viewpoints; overlapping instances or unclear counting conventions; and descriptions that require external knowledge and invite languageprior judgments rather than image-grounded verification. These settings can affect mutation validity, verifier diagnosis, or repair quality. A localized edit may fail to create an unambiguous mismatch, a Verifier may misidentify the relevant evidence, or a Repairer may alter additional unsupported details. The independent audit evaluates these stages separately rather than treating final acceptance as sufficient evidence of a faithful trajectory.

Task extensions. The loop can be adapted when a task provides a known-correct multimodal instance and a structured operation that creates a verifiable error. For VQA, the Mutator could alter an answer, rationale, or question condition and the Repairer could recover the visually supported answer. For visual grounding, it could perturb a region, bounding box, or referring phrase; for referring expressions, it could alter target attributes or relations. The present experiments establish caption-based verification and benchmark transfer, not task-specific self-play training for these extensions.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

## Answer: [Yes]

Justification: The abstract and introduction state the main contribution of PROVE as a verifier-guided self-play framework for improving vision-language verifiers. The claims are supported by the method description, the main compositional understanding results, the broader benchmark results, and the limitations discussed in Section 4.3.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Section 4.3 discusses the limitations of the current action space, difficulty saturation, general-VL regression at small scale, and the reliance on clean reference captions.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [N/A]

Justification: The paper does not present theoretical results or formal theorems. The contribution is an empirical self-play training framework for vision-language verifiers.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

## Answer: [No]

Justification: Sections 3 and 4 and Appendix A describe the method, data size, benchmarks, optimization settings, and reference compute. Appendix F gives the complete role prompts, output formats, reward weights, and cost definitions. Some supplemental audit/baseline settings are not supplied in the current materials; full reproduction requires these details and the planned code release.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [No]

Justification: The training uses existing image-caption data and public benchmarks. The complete prompts are listed in Appendix F; training and evaluation code and launch scripts are planned for release but are not included in the present supplement.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

## Answer: [No]

Justification: The backbone, data sizes, principal optimization settings, evaluation metrics, and matched comparisons are specified, and Appendix F gives the full prompt templates, reward weights, and cost definitions. Supplemental audit/baseline settings remain needed for complete reproduction.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

## Answer: [Yes]

Justification: Appendix C.3 reports means and standard deviations across three independent Qwen3.5-2B training runs for the core compositional average and five broader benchmarks. These measure training-run variability, not confidence intervals or significance tests.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

## Answer: [Yes]

Justification: Section 4 and Appendix A report that the main experiments use a single node with 8 NVIDIA A100 GPUs, BF16 mixed precision, rollout size, batch size, and optimizer settings. Appendix A.4 reports wall-clock time, GPU-hours, and generated-token counts for the reference run and supervision baselines.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

## Answer: [Yes]

Justification: The work studies verifier training for image-text understanding using existing datasets, public benchmarks, and standard VLM training procedures. We are not aware of any aspect of the research that violates the NeurIPS Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses verifier training for detecting image-text mismatches and hallucinations, which can improve reliability of multimodal systems. Potential risks include over-reliance on imperfect verifiers or use of stronger verification models in automated filtering systems; these concerns are partly addressed through the limitations discussion.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [N/A]

Justification: The paper does not introduce a new high-risk pretrained model, image generator, or scraped dataset. It trains a verifier using existing image-caption data and evaluates on public benchmarks; any released code will be for reproducing the training and evaluation pipeline.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The paper cites the Qwen3.5 backbone, VeRL/HybridFlow framework, ShareGPT-4o-Image-style data, and all evaluation benchmarks used in the experiments. We will follow the licenses and terms of use of these existing assets when releasing code and reproduction instructions.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [No]

Justification: The current supplement documents the training framework and experiments; the proposed training and evaluation code artifact and its complete reproduction documentation are not yet included.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [N/A]

Justification: The paper does not involve crowdsourcing experiments or research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [N/A]

Justification: The paper does not involve human subjects research, crowdsourcing, or user studies, so IRB approval is not applicable.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: The core method uses a multimodal LLM backbone as the Mutator, Verifier, and Repairer under different role prompts. This usage is central to the method and is described in Sections 3 and 4.