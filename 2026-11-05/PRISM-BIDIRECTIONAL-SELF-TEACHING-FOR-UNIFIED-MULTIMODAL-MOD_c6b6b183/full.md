# PRISM: BIDIRECTIONAL SELF-TEACHING FOR UNIFIED MULTIMODAL MODELS

Peng Zhang<sup>1,2</sup>, Qian Chen<sup>2</sup>, Xianyin Zhang<sup>2</sup>, Yimin Tian<sup>2</sup>, Ruite Fei<sup>1</sup>, Dongyi Lv<sup>1</sup>, Lifan Guo<sup>2†</sup>, Chi Zhang<sup>2</sup>, and Shuai Liu<sup>1∗</sup>

<sup>1</sup>Xi’an Jiaotong University <sup>2</sup>Qwen DianJin Team, Alibaba Cloud Computing

## ABSTRACT

Unified Multimodal Models (UMMs) aim to support visual understanding and generation within a single model. Existing post-training methods either use understanding as a fixed teacher to improve generation, or rely on ground-truth trajectories to supervise both understanding and generation. Prism instead derives bidirectional supervision from self-generated interleaved rollouts, constructing training targets according to the reliability of their text and image chains. We propose Prism, a bidirectional self-teaching post-training method for UMMs. Like a prism decomposing mixed light, Prism decouples a coupled interleaved trajectory into a text chain and an image chain, assesses the reliability of each chain, and constructs a corresponding training target from their reliability states. Experiments on both SenseNova-U1 and BAGEL show that Prism improves both understanding and generation. On the generation side, GenEval improves from 85.8 to 89.2 (+3.4), and DPG improves from 86.1 to 87.3 (+1.2) on SenseNova-U1. On the understanding side, the total MME score improves from 2282.6 to 2311.6, and MMVP improves from 58.0 to 62.0. Consistent improvements are observed on BAGEL. These results highlight that Prism enables coordinated improvements in generation and understanding by constructing reliability-guided self-training targets from self-generated trajectories, without relying on an external verifier or externally annotated corrected trajectories during target construction. Code is available at LinkLinkLinkLinkLinkLinkLink.

I want to learn how to cook tomato and egg stir-fry. Please give me a beginner-friendly illustrated tutorial.  
![](images/a01347b28a4d68386f41b1e8aa358d43152dcf91b9f9a7f4b1e59a9cb10c33b6.jpg)  
Base

![](images/6782881be91bce9ea62546ba1759f136af2e495fe0b530500fa30a71110acc44.jpg)

![](images/28baf1b0b0787df2fef732ac021b97d817301f8db02e4f4c0a89146cae217d05.jpg)  
Figure 1: Capabilities of Prism. Prism improves both understanding and generation through selfteaching on interleaved text-image trajectories. The base model shows redundant steps on step 2 in the cooking tutorial, while Prism produces a coherent and correct tutorial. It also improves performance on standalone understanding and generation tasks.

## 1 INTRODUCTION

With the rapid development of large language models (Brown et al., 2020; Sun et al., 2021; Bai et al., 2023a) and multimodal generation models (Liu et al., 2023; Chen et al., 2024b; Bai et al., 2023b), researchers have begun to explore unifying understanding and generation within a single model. Unlike treating understanding and generation as two separate models, Unified Multimodal Models (UMMs) integrate text, images, and their interactions into a unified architecture, enabling visual question answering and image generation to share knowledge and training signals. Recent UMMs, from Chameleon (Team, 2024) to SenseNova-U1 (Diao et al., 2026), have demonstrated the feasibility of supporting both understanding and generation within a single model through different architectural designs. However, these models also reveal a fundamental problem: the understanding and generation capabilities within a UMM are asymmetric.

![](images/f5405fa5cb41050927036b8a0958c2461995cc4bdb436ca00b3434c25cba0e20.jpg)  
Figure 2: Difference between Prism and previous post-training methods. (a) Some prior methods are data-driven or rely on external supervision or verifiers. (b) Others improve generation through understanding in a fixed direction. (c) Prism instead enables bidirectional improvement between understanding and generation without external supervision or verifiers.

Recent studies reveal an asymmetry in UMMs: a model can judge whether an image matches a text, yet fail to generate an image that follows the same text (Xie et al., 2026a). Most methods exploit this in a fixed direction, using understanding to improve generation. GvU (Pan et al., 2026) and SRUM (Jin et al., 2025) turn alignment judgments into generation rewards, Internal Gap Self-Improvement (Han et al., 2026) turns them into supervision or DPO preference pairs, and RecA (Xie et al., 2026a) reconstructs images from visual understanding embeddings. Others work in reverse, generating depth, segmentation, or reconstruction targets to sharpen perception (Su et al., 2026), or fine-tuning on synthesized editing and interleaved reasoning traces (Zheng et al., 2026; Gu et al., 2026). A further set applies preference or policy optimization (Yang et al., 2026; Nie et al., 2026; Hu et al., 2026; Yan et al., 2026), and UniCorn (Fang et al.,

![](images/03f371a896e040b184acbb36d663e9bae8aaf95d6d6752a4add99ec4f8ffe140.jpg)  
Figure 3: Prism’s position by ground-truth supervision and improved capabilities.

2026) explores self-play. These methods either are limited to improving generation or rely on external ground-truth pairs, reward models, or verifiers. Prism instead derives bidirectional supervision from self-generated interleaved trajectories, building training targets from text and image reliability without ground-truth trajectories or an external verifier, as illustrated in Figure 2 and Figure 3.

Prism improves both understanding and generation through self-teaching on interleaved text-image tasks. In these trajectories, text and image alternate across multiple steps rather than forming a single input-output pair. Because the text and image chains are coordinated expressions of the same underlying task, each chain can be audited independently. An interleaved trajectory can therefore be decomposed into a text chain that represents understanding and an image chain that represents generation, each of which may be reliable or unreliable in different ways.

As shown in Figure 1, when the base model generates an interleaved cooking tutorial, the first image already shows chopped tomatoes during the ingredient-preparation stage, while the next step again depicts cutting tomatoes. For tutorial-style tasks, this is either redundant or indicates leakage of a future state into an earlier step. Conversely, although not shown in Figure 1, the opposite case can also occur: the image chain may still complete the task correctly even when the text chain contains erroneous reasoning or descriptions, because the image does not strictly follow the faulty text.

Therefore, the key challenge is to identify, for each sample, whether the understanding or generation side is reliable enough to teach the other. Prism addresses this challenge by performing sample-wise self-teaching over interleaved trajectories, enabling bidirectional improvement in both understanding and generation.

We propose Prism, a bidirectional self-teaching post-training method for interleaved text-image trajectories. Prism treats each self-sampled trajectory as a coupled multimodal signal and decomposes it into two independently assessable views: a text chain and an image chain. By comparing the reliability of these two chains, Prism avoids assuming a fixed supervision direction and instead determines on a per-sample basis whether text should supervise image, image should supervise text, or unreliable patterns in both chains should be suppressed.

The key contributions can be summarized as follows:

• Method: We introduce a bidirectional post-training paradigm that selects the teaching direction for each trajectory, allowing multimodal understanding and generation to improve each other rather than imposing a fixed supervision direction.

• Framework: Prism provides a reliability-guided self-teaching framework. It constructs training targets from self-generated interleaved rollouts by assessing the reliability of their text and image chains, without ground-truth supervision or external verifiers.

• Performance: Prism improves multimodal understanding and generation across different UMMs, including interleaved generation. The results demonstrate its effectiveness and generality.

## 2 RELATED WORK

## 2.1 UNIFIED UNDERSTANDING AND GENERATION MODELS

UMM architectures can be broadly categorized into four types. Bridge architectures (Chen et al., 2025a; Wu et al., 2025b; Song et al., 2026) retain separate understanding and generation backbones connected through lightweight interfaces. Unified Transformer architectures (Team, 2024; Zhou et al., 2025a; Xie et al., 2025; Wang et al., 2024; Wu et al., 2025a; Chen et al., 2025b; Xie et al., 2026b) process both modalities within a shared Transformer backbone, with variants using discrete image tokens, mixed autoregressive-diffusion objectives, or decoupled visual encoders. Mixtureof-Transformer-Experts (MoT) architectures (Deng et al., 2025; Diao et al., 2026) allocate separate expert parameters for understanding and generation while sharing attention. Diffusion-based architectures (AI et al., 2026) employ diffusion language models to jointly model text and images.

## 2.2 UMM POST-TRAINING

Existing UMM post-training methods follow four main directions. Preference optimization and reinforcement learning use externally scored pairs in HermesFlow (Yang et al., 2026), reward components in Unified Interleaved GRPO (Nie et al., 2026) and BRAID (Hu et al., 2026), or ground-truth trajectories for VLM adaptation in S-GRPO (Yan et al., 2026). Data-construction approaches include Uni-Edit (Zheng et al., 2026), with 148K editing examples, and ThinkMorph (Gu et al., 2026), with approximately 24K reasoning traces. Specialized generation tasks include RecA (Xie et al., 2026a), which primarily improves generation through image reconstruction, and UniMRG (Su et al., 2026), which enhances perception using externally produced depth and segmentation targets. Self-judging methods use understanding-side feedback to guide generation, including GvU (Pan et al., 2026), SRUM (Jin et al., 2025), and Internal Gap Self-Improvement (Han et al., 2026). UniCorn (Fang et al., 2026) constructs supervision through self-play. UniRL (Mao et al., 2025) improves both tasks using self-generated images, but requires pre-constructed question-answer pairs as ground-truth supervision and applies a fixed round-trip via SFT and GRPO.

## 3 METHOD

Unlike prior post-training methods that rely on external supervision or a fixed supervision direction, Prism derives bidirectional supervision from self-generated interleaved trajectories. It constructs training targets according to the reliability of text and image chains, without ground-truth trajecto ries or external verifiers. As illustrated in Figure 4, given a task prompt, the model generates an interleaved trajectory. Frozen model copies then assess the text and image chains and map the trajectory to one of four training actions. Only the rollout model is updated using the resulting targets.

![](images/710a9dbebf7e8e05e228355caab25a41f699de8e23159da20627195796164aab.jpg)  
${ r _ { \mathrm { i m } g } / \ r _ { \mathrm { f } x \mathrm { f } } } ;$ denote the reason for the image / text error, respectively. $\mathsf { P } _ { r e c }$ : the instruction for recovering T from image I. UMM (・) : · denotes the UMM branch in use. If empty, both branches are used. Trainable Frozen  
Figure 4: Overview of the Prism. Given a text prompt, Prism first uses the UMM to roll out an interleaved trajectory. It then uses the UMM in a judging role to assess the reliability of the text chain and image chain. Based on the resulting status, Prism selects the corresponding processing action and loss function. All UMM instances are initialized from the same base model. Only the rollout UMM is trainable, while the UMMs used for judging and action selection remain frozen.

## 3.1 PRELIMINARY

Given a text prompt P, the UMM generates an interleaved text-image trajectory:

$$
\tau = ( T _ { 1 } , I _ { 1 } , T _ { 2 } , I _ { 2 } , \ldots , T _ { K } , I _ { K } ) ,\tag{1}
$$

where $T _ { i }$ and $I _ { i }$ denote the text and image segments at step i. For the interleaved tasks considered here, the trajectory has two modality-specific views, the complete text chain $T = ( T _ { 1 } , \dots , T _ { K } )$ representing understanding through planning and reasoning, and the complete image chain $I \ =$ $( \bar { I _ { 1 } } , \ldots , \bar { I _ { K } } )$ , representing generation through visual states and cross-step consistency. Both are projections of the same interleaved trajectory.

Standard UMM interleaved training models the text-image trajectory autoregressively:

$$
p _ { \theta } ( \tau \mid P ) = \prod _ { i = 1 } ^ { K } p _ { \theta } ( T _ { i } \mid P , T _ { < i } , I _ { < i } ) ~ p _ { \theta } ( I _ { i } \mid P , T _ { \leq i } , I _ { < i } ) .\tag{2}
$$

The standard training loss for an interleaved trajectory is

$$
\mathcal { L } _ { \mathrm { i n t e r l e a v e } } ( \tau ) = \sum _ { i = 1 } ^ { K } [ \lambda _ { \mathrm { u n d } } \mathcal { L } _ { \mathrm { u n d } } ( f _ { \theta } ( \cdot  { | } P , T _ { < i } , I _ { < i } ) , T _ { i } ) + \lambda _ { \mathrm { g e n } } \mathcal { L } _ { \mathrm { g e n } } ( f _ { \theta } ( \cdot  { | } P , T _ { \le i } , I _ { < i } ) , I _ { i } ) ]\tag{3}
$$

where ${ \mathcal { L } } _ { \mathrm { u n d } }$ is the token-level cross-entropy loss for text prediction, and ${ \mathcal { L } } _ { \mathrm { g e n } }$ is the flow-matching loss for image generation (Lipman et al., 2022). $\lambda _ { \mathrm { u n d } }$ and $\lambda _ { \mathrm { g e n } }$ are modality-balancing weights that may vary across architectures.

## 3.2 PER-MODALITY RELIABILITY JUDGMENT

The core premise of Prism is that, under the same task prompt $P ,$ , the text chain $T$ and the image chain I are two modality-specific expressions of the same underlying task content. Based on this premise, Prism judges the reliability of the complete text chain and complete image chain separately:

$$
( v _ { \mathrm { t x t } } , r _ { \mathrm { t x t } } ) = \mathcal { I } ( P , T , q _ { \mathrm { t x t } } ) , \qquad ( v _ { \mathrm { i m g } } , r _ { \mathrm { i m g } } ) = \mathcal { I } ( P , I , q _ { \mathrm { i m g } } ) ,\tag{4}
$$

where $\mathcal { I }$ is a frozen copy of the base model, and $q _ { \mathrm { t x t } }$ and $q _ { \mathrm { i m g } }$ are review instructions. The indicators v<sub>txt</sub>, $v _ { \mathrm { i m g } } \in \{ 0 , 1 \}$ denote the reliability of the understanding and generation sides, while $r _ { \mathrm { t x t } }$ and $r _ { \mathrm { i m g } }$ specify detected errors. Prism judges the complete chains and uses the resulting states to select a training action, as illustrated in Figure 4.

## 3.3 RELIABILITY-GUIDED SUPERVISION CONSTRUCTION

Based on the judgments in Section 3.2, Prism selects one of four actions for each trajectory: enhancing cross-modal consistency for reliable trajectories, rewriting an unreliable chain from the reliable side, or suppressing trajectories whose two chains are unreliable.

Both pass, $v _ { \mathrm { t x t } } = 1 , v _ { \mathrm { i m g } } = 1$ . When both chains are reliable, the trajectory provides a reliable interleaved training sample. Prism conditions text reconstruction on the image chain, reinforcing cross-modal consistency throughout the trajectory. Let $P _ { \mathrm { r e c } }$ denote the instruction for recovering $\bar { T }$ from image chain I under the original task. The objective is

$$
\mathcal { L } _ { \mathrm { p a s s } } = \mathcal { L } _ { \mathrm { u n d } } \big ( f _ { \theta } ( T \mid P _ { \mathrm { r e c } } , I ) , T \big ) .\tag{5}
$$

Text-to-image rewriting, $v _ { \mathrm { t x t } } ~ = ~ 1 , v _ { \mathrm { i m g } } ~ = ~ 0 .$ . When the text chain is judged reliable but the image chain is not, Prism preserves $T$ and uses P, T, and $r _ { \mathrm { i m g } }$ to generate a corrected image chain $I ^ { + } = ( I _ { 1 } ^ { + } , \ldots , I _ { K } ^ { + } )$ . The resulting corrected trajectory and training objective are

$$
\tau _ { \mathrm { t 2 i } } ^ { + } = ( T _ { 1 } , I _ { 1 } ^ { + } , T _ { 2 } , I _ { 2 } ^ { + } , \dots , T _ { K } , I _ { K } ^ { + } ) , \qquad \mathscr { L } _ { \mathrm { t 2 i } } = \mathscr { L } _ { \mathrm { i n t e r l e a v e } } ( \tau _ { \mathrm { t 2 i } } ^ { + } ) .\tag{6}
$$

Image-to-text rewriting, $v _ { \mathrm { t x t } } = 0 , v _ { \mathrm { i m g } } = 1$ . When the image chain is judged reliable but the text chain is not, Prism preserves I and uses $P , I ,$ and $r _ { \mathrm { t x t } }$ to generate a corrected text chain $T ^ { + } =$ $( T _ { 1 } ^ { + } , \dots , T _ { K } ^ { + } )$ . The resulting corrected trajectory and training objective are

$$
\tau _ { \mathrm { i 2 t } } ^ { + } = ( T _ { 1 } ^ { + } , I _ { 1 } , T _ { 2 } ^ { + } , I _ { 2 } , \ldots , T _ { K } ^ { + } , I _ { K } ) , \qquad \mathcal { L } _ { \mathrm { i 2 t } } = \mathcal { L } _ { \mathrm { i n t e r l e a v e } } ( \tau _ { \mathrm { i 2 t } } ^ { + } ) .\tag{7}
$$

Both fail, $v _ { \mathrm { t x t } } = 0 , v _ { \mathrm { i m g } } = 0 .$ . When both chains are judged unreliable, Prism does not construct a positive target. Instead, it applies a token-level unlikelihood objective to the text tokens in the original trajectory, reducing the probability of reproducing its erroneous text patterns:

$$
\mathcal { L } _ { \mathrm { f a i l } } ( \tau ) = - \sum _ { i = 1 } ^ { K } \sum _ { n = 1 } ^ { | T _ { i } | } \log \left( 1 - p _ { \theta } \big ( t _ { i , n } ~ | ~ P , T _ { < i } , I _ { < i } , t _ { i , < n } \big ) \right) ,\tag{8}
$$

where $T _ { i } = ( t _ { i , 1 } , \dots , t _ { i , | T _ { i } | } )$ denotes the i-th text segment and $t _ { i , < n }$ denotes the tokens preceding $t _ { i , n }$ within that segment. Unlike text, image generation is optimized through continuous flow matching, for which no direct counterpart of token-level unlikelihood is available. We therefore restrict negative supervision to the text tokens.

For the two rewriting cases, since the rewritten chain is newly generated and may fail to correct the detected errors, Prism verifies each rewritten trajectory before using it as a positive training target, retrying failed corrections and skipping trajectories that remain invalid. Appendix F details the verification procedure. During training, only the rollout model is updated, while judgment, rewriting, and verification use frozen copies of the base checkpoint to prevent co-evolution with the rollout model. Jointly updating them leads to training collapse, as shown in Appendix D.

## 3.4 TRAINING OBJECTIVE

Each trajectory contributes to one of the four objectives according to its reliability state. So the overall objective is

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { p a s s } } \mathcal { L } _ { \mathrm { p a s s } } + \mathcal { L } _ { \mathrm { t 2 i } } + \mathcal { L } _ { \mathrm { i 2 t } } + \lambda _ { \mathrm { f a i l } } \mathcal { L } _ { \mathrm { f a i l } } ,\tag{9}
$$

where $\lambda _ { \mathrm { p a s s } } , \lambda _ { \mathrm { i 2 t } }$ , and $\lambda _ { \mathrm { f a i l } }$ weight the both-pass, image-to-text rewriting, and both-fail objectives, respectively. The text-to-image rewriting objective uses unit weight.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Data. We use GPT-5.5 to generate 2.4K task prompt seeds spanning tutorials, stories, spatial reasoning, and other task types. These prompts provide task descriptions only. Prism generates interleaved trajectories online and constructs training targets from the model’s own rollouts.

Evaluation. We evaluate image generation on GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024), multimodal understanding on MME (Fu et al., 2026), MMBench (Liu et al., 2024),

Table 1: Prism improves image generation performance.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Prism</td><td colspan="7">GenEval</td><td rowspan="2">DPG</td></tr><tr><td>Single</td><td>Two</td><td>Count</td><td>Color</td><td>Position</td><td>Attri.</td><td>Overall</td></tr><tr><td>SenseNo</td><td>x</td><td>100.0</td><td>92.9</td><td>75.6</td><td>91.7</td><td>85.5</td><td>68.7</td><td>85.8</td><td>86.1</td></tr><tr><td>va-U1</td><td>√</td><td> $\mathbf { 1 0 0 . 0 } _ { ( 0 . 0 ) }$ </td><td> $\mathbf { 9 6 . 7 } _ { ( + 3 . 8 ) }$ </td><td> ${ \mathbf 8 7 . 2 } _ { ( + 1 1 . 6 ) }$ </td><td> $\mathbf { 8 9 . 6 } _ { ( - 2 . 1 ) }$ </td><td> ${ \mathbf 8 6 . 8 } _ { ( + 1 . 3 ) }$ </td><td> $7 5 . 0 _ { ( + 6 . 3 ) }$ </td><td> $\mathbf { 8 9 . } 2 _ { ( + 3 . 4 ) }$ </td><td> ${ \mathbf 8 7 . 3 } _ { ( + 1 . 2 ) }$ </td></tr><tr><td rowspan="2">BAGEL</td><td>x</td><td>99.1</td><td>93.0</td><td>76.5</td><td>86.0</td><td>47.2</td><td>57.0</td><td>76.5</td><td>84.0</td></tr><tr><td>√</td><td> $9 9 . 4 _ { ( + 0 . 3 ) }$ </td><td> $9 6 . 5 _ { ( + 3 . 5 ) }$ </td><td> $7 6 . 9 _ { ( + 0 . 4 ) }$  </td><td> $\mathbf { 8 6 . 5 } _ { ( + 0 . 5 ) }$ </td><td> ${ \bf 4 8 . 0 } _ { ( + 0 . 8 ) }$ </td><td> ${ \bf 6 2 . 5 } _ { ( + 5 . 5 ) }$ </td><td> $7 8 . 3 _ { ( + 1 . 8 ) }$ </td><td> $\mathbf { 8 5 . 0 } _ { ( + 1 . 0 ) }$ </td></tr></table>

Table 2: Prism improves multimodal understanding performance. We evaluate the effectiveness of Prism across understanding benchmarks covering general knowledge, subject knowledge, and others.
<table><tr><td>Model</td><td>Prism</td><td>MME-P</td><td>MME-C</td><td>MMBench</td><td>MMMU</td><td>MMStar</td><td>MMVP</td></tr><tr><td rowspan="2">SenseNova-U1</td><td>x</td><td>1699.7</td><td>582.9</td><td>86.1</td><td>59.0</td><td>65.1</td><td>58.0</td></tr><tr><td>√</td><td> $1 7 1 3 . 4 _ { ( + 1 3 . 7 ) }$ </td><td> $5 9 8 . 2 _ { ( + 1 5 . 3 ) }$ </td><td> ${ \mathbf 8 7 . 3 } _ { ( + 1 . 2 ) }$ </td><td> ${ \bf 6 0 . 6 } _ { ( + 1 . 6 ) }$ </td><td> ${ \bf 6 6 . 5 } _ { ( + 1 . 4 ) }$ </td><td> $\mathbf { 6 } 2 . \mathbf { 0 } _ { ( + 4 . 0 ) }$ </td></tr><tr><td rowspan="2">BAGEL</td><td>x</td><td>1681.0</td><td>679.6</td><td>84.6</td><td>53.7</td><td>65.0</td><td>69.3</td></tr><tr><td>√</td><td> $\mathbf { 1 6 8 5 . 2 } _ { ( + 4 . 2 ) }$ </td><td> ${ \bf 6 8 5 . 4 } _ { ( + 5 . 8 ) }$ </td><td> ${ \mathbf 8 8 . 4 } _ { ( + 3 . 8 ) }$ </td><td> ${ \pmb 5 } 5 . { \pmb 0 } _ { ( + 1 . 3 ) }$ </td><td> ${ \bf 6 9 . 6 } _ { ( + 4 . 6 ) }$ </td><td> $7 0 . 0 _ { ( + 0 . 7 ) }$ </td></tr></table>

Table 3: OpenING pairwise evaluation. Left: Win rates under different tie-handling settings. Right: Zero-image rates, i.e., outputs with no generated image. Lower is better.
<table><tr><td>Method</td><td>FDT</td><td>w/o Tie</td><td> ${ \bf w } / \mathrm { T i e } \left( { \bf 0 } \right)$ </td><td>w/ Tie (.5)</td></tr><tr><td>SenseNova-U1 + Prism</td><td>74.93%</td><td>76.65%</td><td>66.79%</td><td>73.22%</td></tr><tr><td>SenseNova-U1</td><td>72.03%</td><td>72.91%</td><td>62.55%</td><td>69.66%</td></tr><tr><td>GPT-40+DALL-E3</td><td>63.79%</td><td>66.92%</td><td>56.36%</td><td>64.25%</td></tr><tr><td>Human</td><td>61.10%</td><td>64.66%</td><td>51.90%</td><td>61.76%</td></tr><tr><td>Gemini1.5+Flux</td><td>38.35%</td><td>39.04%</td><td>31.23%</td><td>41.24%</td></tr><tr><td>BAGEL + Prism</td><td>34.70%</td><td>31.66%</td><td>25.80%</td><td>35.05%</td></tr><tr><td>BAGEL</td><td>30.84%</td><td>27.94%</td><td>23.47%</td><td>31.47%</td></tr><tr><td>SEED-X</td><td>21.27%</td><td>17.42%</td><td>15.92%</td><td>20.23%</td></tr></table>

![](images/1f281743f46aefd517c7c3e9f87d7721adbd754a96b28ef30bb9c6eb67035135.jpg)

MMMU (Yue et al., 2024), MMStar (Chen et al., 2024a), MMVP (Tong et al., 2024), and POPE (Li et al., 2023), and interleaved generation on OpenING (Zhou et al., 2025b). We follow the official evaluation settings for all benchmarks and report the multiple-choice accuracy for MMMU.

Training. Since some UMMs lack interleaved text-image generation support or publicly available model weights, we select SenseNova-U1 (Diao et al., 2026) and BAGEL (Deng et al., 2025), which satisfy both requirements. The main experiments use a learning rate of 3e-5, with $\lambda _ { \mathrm { p a s s } } = 0 . 5$ , and $\lambda _ { \mathrm { f a i l } } = 0 . 0 0 5$ . The text-to-image rewriting objective uses unit weight. Training is performed on 16 A800 GPUs for approximately 24 hours.

For more details, please refer to the Appendix A.

## 4.2 PRISM SIMULTANEOUSLY IMPROVES UNDERSTANDING AND GENERATION

We evaluate Prism on SenseNova-U1 and BAGEL across image generation and multimodal understanding benchmarks. Tables 1 and 2 show consistent gains on both models.

On SenseNova-U1, Prism improves GenEval by 3.4 points and DPG by 1.2 points, with notable gains in counting, two-object generation, and attribute binding. On understanding benchmarks, it improves MME by 29.0 points and MMVP by 4.0 points, with gains on the other benchmarks as well. BAGEL shows the same overall trend on the benchmarks reported in Tables 1 and 2, providing evidence that the method can transfer across the two evaluated UMM backbones.

![](images/ecf34dd0a8868a1bf2528031542247517075d78db2ed37a654c89131596a3e34.jpg)  
A warm, sunlit rustic kitchen still life: … a small ceramic bowl and a wooden cutting board with a knife. … whole scene is bright, inviting, and full of rustic charm, photorealistic food photography.  
…, pink petals drifting down onto a wicker basket and a bottle of lemonade. … The tree is heavy with blossoms, its branches bending … Sunlight filters through the blossoms, casting a gentle pink glow over the scene…  
A cheerful toy room scene on a soft carpet: a red toy car in the front of a blue toy train, ..., and a picture book lying on the table. A window with sunshine is on the left wall, and the room is bright and colorful.

Figure 5: Improved generation. We use SenseNova-U1 as the baseline model. The model trained with Prism shows more accurate adherence to prompt, including spatial relations, object counts, and style. The underlined text highlights the generation differences under complex instructions.  
Table 4: Compared with similar methods on BAGEL, our method can simultaneously improve both understanding and generation, demonstrating bidirectional enhancement capability.
<table><tr><td>Model</td><td>MME</td><td>MMBench</td><td>MMMU</td><td>MMStar</td><td>MMVP</td><td>GenEval</td><td>DPG</td></tr><tr><td>BAGEL</td><td>2361</td><td>84.6</td><td>53.7</td><td>65.0</td><td>69.3</td><td>76.5</td><td>84.0</td></tr><tr><td>+ SRUM</td><td> $2 3 5 0 _ { ( - 1 1 ) }$ </td><td> $8 4 . 4 ( . 0 . 2 )$ </td><td> $5 3 . 4 ( . 0 . 3 )$ </td><td> $6 4 . 7 _ { ( - 0 . 3 ) }$ </td><td> $7 0 . 0 _ { ( + 0 . 7 ) }$ </td><td> $7 7 . 4 ( + 0 . 9 ) $ </td><td> $8 5 . 2 _ { ( + 1 . 2 ) }$ </td></tr><tr><td>+ UniCorn</td><td> $2 3 3 7 ( - 2 4 )$ </td><td> $8 4 . 1 _ { ( - 0 . 5 ) }$ </td><td> $5 4 . 5 _ { ( + 0 . 8 ) }$ </td><td> $6 5 . 0 _ { ( + 0 . 0 ) }$ </td><td> $7 0 . 0 _ { ( + 0 . 7 ) }$ </td><td> $8 0 . 3 _ { ( + 3 . 8 ) }$ </td><td> $8 6 . 2 _ { ( + 2 . 2 ) }$ </td></tr><tr><td>+ Prism</td><td> $2 3 7 1 _ { ( + 1 0 ) }$ </td><td> $8 8 . 4 _ { ( + 3 . 8 ) }$ </td><td> $5 5 . 0 _ { ( + 1 . 3 ) }$ </td><td> $6 9 . 6 _ { ( + 4 . 6 ) }$ </td><td> $7 0 . 0 _ { ( + 0 . 7 ) }$ </td><td> $7 8 . 3 _ { ( + 1 . 8 ) }$  </td><td> $8 5 . 0 _ { ( + 1 . 0 ) }$ </td></tr></table>

## 4.3 PRISM IMPROVES INTERLEAVED GENERATION

We evaluate Prism on OpenING Zhou et al. (2025b), which measures open-ended interleaved image–text generation with pairwise win rates. Table 3 reports the win rates and zero-image rates.

Prism improves SenseNova-U1 by 2.90 to 4.24 points and BAGEL by 2.33 to 3.86 points across all tie-handling settings. Among the compared entries, SenseNova-U1 with Prism achieves the highest win rate in all four settings. Prism also reduces the zero-image rate of BAGEL from 30.7% to 14.8%, indicating improved ability to generate interleaved trajectories. SenseNova-U1 already has a near-zero zero-image rate of 0.7% before training, and Prism preserves this capability with a rate of 0.9%. Overall, these results show that Prism improves interleaved generation across two UMMs.

## 4.4 COMPARISON WITH RELATED POST-TRAINING METHODS

We compare Prism with SRUM and UniCorn on BAGEL in Table 4. SRUM and UniCorn both improve GenEval and DPG, but reduce MME and MMBench. On other understanding benchmarks, some scores stay flat or drop. This shows that they do not achieve consistent gains in both generation and multimodal understanding. By contrast, Prism improves MME by 10 points, MMBench by 3.8 points, MMMU by 1.3 points, and MMStar by 4.6 points, while maintaining comparable MMVP performance. It also improves GenEval by 1.8 points and DPG by 1.0 point. Results show the four losses contribute distinctly to understanding and generation.

## 4.5 ABLATION EXPERIMENT

To ablate each loss, we train trajectories assigned to that loss on their unmodified rollouts with $\mathcal { L } _ { \mathrm { i n t e r l e a v e } } ,$ leaving the other losses unchanged. Table 5 shows that replacing $\mathcal { L } _ { \mathrm { t 2 i } }$ reduces GenEval by 1.2 points, while replacing ${ \mathcal { L } } _ { \mathrm { i 2 t } }$ reduces MMVP by 5.3 points. Replacing $\mathcal { L } _ { \mathrm { p a s s } }$ slightly improves

![](images/40f966a2dc66fff83ca827c1b55f976bb78bd04759d52f00b2fba020eb1b1487.jpg)  
Figure 6: Improved understanding. We use SenseNova-U1 as the baseline model. The model trained with Prism demonstrates more accurate responses on diverse multimodal multiple-choice tasks, including code reasoning, visual counting, numerical reasoning, and commonsense understanding. For more examples, please refer to the Appendix G

Table 5: Loss ablations on SenseNova-U1. Dashes denote the base model without post-training. Crosses denote replacement with $\mathcal { L } _ { \mathrm { i n t e r l e a v e } }$ on the corresponding unmodified rollouts.
<table><tr><td colspan="4">Loss Terms</td><td colspan="5">Understanding</td><td colspan="2">Generation</td></tr><tr><td> $\mathcal { L } _ { \mathrm { p a s s } } \mathcal { L } _ { \mathrm { t 2 i } } \mathcal { L } _ { \mathrm { i 2 t } } \mathcal { L } _ { \mathrm { f a i l } }$ </td><td></td><td></td><td></td><td>MME</td><td>MMBench</td><td>MMMU</td><td>MMStar</td><td>MMVP</td><td>GenEval</td><td>DPG</td></tr><tr><td>-</td><td>1</td><td>1</td><td>1</td><td>2283</td><td>86.1</td><td>59.0</td><td>65.1</td><td>58.0</td><td>85.8</td><td>86.1</td></tr><tr><td>x</td><td></td><td>VVXV√</td><td>VVVXV</td><td> $2 2 9 9 _ { ( + 1 6 ) }$ </td><td> $8 6 . 6 _ { ( + 0 . 5 ) }$ </td><td> $6 1 . 0 _ { ( + 2 . 0 ) }$ </td><td> $6 5 . 3 _ { ( + 0 . 2 ) }$ </td><td> $6 0 . 0 _ { ( + 2 . 0 ) }$ </td><td> $8 9 . 4 _ { ( + 3 . 6 ) }$ </td><td> $8 7 . 2 _ { ( + 1 . 1 ) }$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $2 2 9 4 ( + 1 1 )$ </td><td> $8 7 . 4 ( + 1 . 3 )$ </td><td> $6 0 . 1 ( + 1 . 1 ) $ </td><td> $6 4 . 9 ( - 0 . 2 ) $ </td><td> $5 9 . 3 _ { ( + 1 . 3 ) }$ </td><td> $8 8 . 0 _ { ( + 2 . 2 ) }$ </td><td> $8 7 . 0 _ { ( + 0 . 9 ) }$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $2 2 9 9 _ { ( + 1 6 ) }$ </td><td> $8 6 . 6 _ { ( + 0 . 5 ) }$ </td><td> $6 0 . 1 _ { ( + 1 . 1 ) }$ </td><td> $6 5 . 6 _ { ( + 0 . 5 ) }$ </td><td> $5 6 . 7 _ { ( - 1 . 3 ) }$ </td><td> $8 8 . 8 _ { ( + 3 . 0 ) }$ </td><td> ${ \mathbf 8 7 . 3 } _ { ( + 1 . 2 ) }$ </td></tr><tr><td></td><td></td><td></td><td></td><td> $2 3 1 0 _ { ( + 2 7 ) }$ </td><td> $8 6 . 8 _ { ( + 0 . 7 ) }$ </td><td> ${ \bf 6 1 . 3 } _ { ( + 2 . 3 ) }$ </td><td> $6 5 . 3 _ { ( + 0 . 2 ) }$ </td><td> $6 0 . 0 _ { ( + 2 . 0 ) }$ </td><td> $8 8 . 7 _ { ( + 2 . 9 ) }$ </td><td> $8 7 . 1 _ { ( + 1 . 0 ) }$ </td></tr><tr><td>VVVV</td><td>VXVVV</td><td></td><td></td><td> $2 3 1 2 ( + 2 9 )$ </td><td> $8 7 . 3 ( + 1 . 2 ) $ </td><td> $6 0 . 6 ( + 1 . 6 )$ </td><td> ${ \bf 6 6 . 5 } _ { ( + 1 . 4 ) }$ </td><td> ${ \bf 6 2 . 0 } _ { ( + 4 . 0 ) }$ </td><td> $8 9 . 2 ( + 3 . 4 ) $ </td><td> $8 7 . 3 ( + 1 . 2 ) $ </td></tr></table>

Table 6: Analysis of reliability-guided target construction on SenseNova-U1.
<table><tr><td>Setting</td><td>GenEval</td><td>DPG</td><td>MME</td><td>MMB</td><td>MMMU</td><td>MMStar</td><td>MMVP</td></tr><tr><td>Base Model</td><td>85.8</td><td>86.1</td><td>2282.6</td><td>86.1</td><td>59.0</td><td>65.1</td><td>58.0</td></tr><tr><td>Base w/ Self-Training</td><td>87.7</td><td>86.6</td><td>2276.3</td><td>85.5</td><td>60.0</td><td>64.7</td><td>58.0</td></tr><tr><td>Prism w/o Rewriting</td><td>88.4</td><td>86.5</td><td>2300.6</td><td>86.8</td><td>61.0</td><td>65.6</td><td>60.7</td></tr><tr><td>Prism w/ Random Routing</td><td>87.6</td><td>86.5</td><td>2297.5</td><td>86.5</td><td>59.9</td><td>65.5</td><td>59.3</td></tr><tr><td>Prism (Ours)</td><td>89.2</td><td>87.3</td><td>2311.6</td><td>87.3</td><td>60.6</td><td>66.5</td><td>62.0</td></tr></table>

GenEval but lowers four of five understanding metrics. Replacing ${ \mathcal { L } } _ { \mathrm { f a i l } }$ lowers both generation metrics and four of five understanding metrics. The four losses play distinct roles in improving understanding, generation, and cross-modal consistency.

## 4.6 ANALYSIS OF RELIABILITY-GUIDED TARGET CONSTRUCTION

Table 6 compares Prism with three controls. Base w/ Self-Training applies $\mathcal { L } _ { \mathrm { i n t e r l e a v e } }$ to all unmodified rollouts. Prism w/o Rewriting keeps judge-based routing but applies $\mathcal { L } _ { \mathrm { i n t e r l e a v e } }$ to the original rollouts assigned to t2i and i2t, leaving pass and fail unchanged. Prism w/ Random Routing shuffles action assignments within each batch while preserving their counts. Full Prism outperforms selftraining and random routing on all seven metrics, and w/o rewriting on six. These results support the contributions of reliability-guided routing and rewritten targets. More appears in Appendix C.

![](images/972f5054ac2bb0581d82982b82441f7ed9b615768489e769f8dc0e94f3514c98.jpg)  
Figure 7: Qualitative examples generated by the model trained with Prism. The figure illustrates a diverse set of outputs, including realistic photography and artistic rendering, among others.

## 4.7 MORE RESULTS

Qualitative. Figure 5 compares SenseNova-U1 with its Prism-trained variant. Prism more faithfully follows prompts involving object composition, counting, spatial relations, and detailed scene descriptions. Figure 6 further shows improved responses on code reasoning, visual counting, document-based numerical reasoning, and relation understanding. These examples complement the quantitative results on generation and multimodal understanding. More examples with diverse styles and quantities are provided in Figure 7 and Appendix H.

Hallucination analysis. We further evaluate Prism on POPE. As shown in Table 7, Prism improves Accuracy and F1 from 85.2 to 88.0 and from 84.9 to 86.7 on BAGEL, respectively. On SenseNova-U1, it improves Accuracy from 85.6 to 87.5 and F1 from 84.0 to 86.6. These results suggest improved visual grounding and fewer hallucination-related errors.

Table 7: POPE benchmark.
<table><tr><td>Model</td><td>Prism</td><td>Acc.</td><td>F1</td></tr><tr><td rowspan="2">BAGEL</td><td>x</td><td>85.2</td><td>84.9</td></tr><tr><td>√</td><td></td><td>88.0(+2.886.7(+1.8)</td></tr><tr><td rowspan="2">SenseNova-U1</td><td>x</td><td>85.6</td><td>84.0</td></tr><tr><td>√</td><td></td><td>87.5(+1.9 86.6(+2.6)</td></tr></table>

## 5 CONCLUSION AND LIMITATIONS

We propose Prism, a bidirectional self-teaching method that derives targets through text and image reliability. It improves generation, understanding and interleaved generation on UMMs.

Despite these gains, Prism has three limitations. It targets interleaved text-image tasks with externally provided prompt seeds, and applies negative supervision to the complete text chain when both chains fail. Future work will extend Prism to broader multimodal tasks, fully self-sampled training, and localized negative supervision.

## REFERENCES

Inclusion AI, Tiwei Bie, Haoxing Chen, Tieyuan Chen, Zhenglin Cheng, Long Cui, Kai Gan, Zhicheng Huang, Zhenzhong Lan, Haoquan Li, et al. Llada2. 0-uni: Unifying multimodal understanding and generation with diffusion large language model. arXiv preprint arXiv:2604.20796, 2026.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023a.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 1(2):3, 2023b.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. Advances in Neural Information Processing Systems, 38, 2026.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37:27056–27087, 2024a.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025b.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024b.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Haiwen Diao, Penghao Wu, Hanming Deng, Jiahao Wang, Shihao Bai, Silei Wu, Weichen Fan, Wenjie Ye, Wenwen Tong, Xiangyu Fan, et al. Sensenova-u1: Unifying multimodal understanding and generation with neo-unify architecture. arXiv preprint arXiv:2605.12500, 2026.

Zhen Fang, Ruiyan Han, XinYu Sun, Yuchen Ma, Ziheng Wang, Yu Zeng, Zehui Chen, Lin Chen, Wenxuan Huang, Wei-Jie Xu, et al. Unicorn: Towards self-improving unified multimodal models through self-generated supervision. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13638–13669, 2026.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. Advances in Neural Information Processing Systems, 38, 2026.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36: 52132–52152, 2023.

Jiawei Gu, Yunzhuo Hao, Huichen Wang, Linjie Li, Michael Qizhe Shieh, Yejin Choi, Ranjay Krishna, and Yu Cheng. Thinkmorph: Emergent properties in multimodal interleaved chain-ofthought reasoning. In International Conference on Learning Representations, volume 2026, pp. 141405–141447, 2026.

Yujin Han, Hao Chen, Andi Han, Zhiheng Wang, Xinyu Liu, Shiwei Zhang, Difan Zou, et al. Turning internal gap into self-improvement: Promoting the generation-understanding unification in mllms. In International Conference on Learning Representations, volume 2026, pp. 136501– 136534, 2026.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Zican Hu, Xuyang Hu, Yiming Liu, Zuwei Long, Wei Liu, Yunzhuo Hao, Jiawei Gu, Linjie Li, Yu Cheng, Zhenhong Sun, et al. Bridging interleaved multi-modal reasoning as a unified decision process. arXiv preprint arXiv:2607.03748, 2026.

Weiyang Jin, Yuwei Niu, Jiaqi Liao, Chengqi Duan, Aoxue Li, Shenghua Gao, and Xihui Liu. Srum: Fine-grained self-rewarding for unified multimodal models. arXiv preprint arXiv:2510.12784, 2025.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 292–305, 2023.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024.

Weijia Mao, Zhenheng Yang, and Mike Zheng Shou. Unirl: Self-improving unified multimodal models via supervised and reinforcement learning. arXiv preprint arXiv:2505.23380, 2025.

Ming Nie, Chunwei Wang, Jianhua Han, Hang Xu, and Li Zhang. Towards unified multimodal interleaved generation via group relative policy optimization. Advances in Neural Information Processing Systems, 38:5332–5353, 2026.

Jiadong Pan, Liang Li, Yuxin Peng, Yu-Ming Tang, Shuohuan Wang, Yu Sun, Hua Wu, Qingming Huang, and Haifeng Wang. Learning to generate via understanding: Understanding-driven intrinsic rewarding for unified multimodal models. arXiv preprint arXiv:2603.06043, 2026.

Lin Song, Wenbo Li, Guoqing Ma, Wei Tang, Bo Wang, Yuan Zhang, Yijun Yang, Yicheng Xiao, Jianhui Liu, Yanbing Zhang, et al. Joyai-image: Awaking spatial intelligence in unified multimodal understanding and generation. arXiv preprint arXiv:2605.04128, 2026.

Zihan Su, Hongyang Wei, Kangrui Cen, Yong Wang, Guanhua Chen, Chun Yuan, and Xiangxiang Chu. Generation enhances understanding in unified multimodal models via multi-representation generation. arXiv preprint arXiv:2601.21406, 2026.

Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, et al. Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation. arXiv preprint arXiv:2107.02137, 2021.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9568–9578. IEEE, 2024.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

Xinyu Wei, Jinrui Zhang, Zeqing Wang, Hongyang Wei, Zhen Guo, Bairui Li, and Lei Zhang. Tiifbench: How does your t2i model follow your instructions? arXiv preprint arXiv:2506.02161, 2025.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12966–12977. IEEE, 2025a.

Size Wu, Zhonghua Wu, Zerui Gong, Qingyi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661, 2025b.

Ji Xie, trevor darrell, Luke Zettlemoyer, and Xudong Wang. Reconstruction alignment improves unified multimodal models. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 120095–120137, 2026a.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. In International Conference on Learning Representations, volume 2025, pp. 28240–28264, 2025.

Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. Advances in Neural Information Processing Systems, 38:47490–47518, 2026b.

Yuming Yan, Kai Tang, Sihong Chen, Ke Xu, Dan Hu, Qun Yu, and Pengfei Hu. S-grpo: Unified post-training for large vision-language models. arXiv preprint arXiv:2604.16557, 2026.

Ling Yang, Xinchen Zhang, Ye Tian, Shiyi Zhang, Chenming Shang, Minghao Xu, Wentao Zhang, and Bin Cui. Hermesflow: Seamlessly closing the gap in multimodal understanding and generation. Advances in Neural Information Processing Systems, 38:62248–62272, 2026.

Yang Ye, Xianyi He, Zongjian Li, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, Li Yuan, et al. Imgedit: A unified image editing dataset and benchmark. Advances in Neural Information Processing Systems, 38, 2026.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9556–9567, 2024.

Dian Zheng, Manyuan Zhang, Hongyu Li, Hongbo Liu, Kai Zou, Kaituo Feng, and Hongsheng Li. Uni-edit: Intelligent editing is a general task for unified model tuning. arXiv preprint arXiv:2605.21487, 2026.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. In International Conference on Learning Representations, volume 2025, pp. 6446–6469, 2025a.

Pengfei Zhou, Xiaopeng Peng, Jiajun Song, Chuanhao Li, Zhaopan Xu, Yue Yang, Ziyao Guo, Hao Zhang, Yuqi Lin, Yefei He, et al. Opening: A comprehensive benchmark for judging openended interleaved image-text generation. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 56–66. IEEE, 2025b.

## A TRAINING AND EVALUATION DETAILS

## A.1 IMPLEMENTATION DETAILS AND HYPERPARAMETERS

We train all models at a resolution of 1024 × 1024 for one epoch. For the interleaved objective, the cross-entropy and flow-matching loss weights are set to 0.1 : 1 for SenseNova-U1 and 0.25 : 1 for BAGEL. We use AdamW with a learning rate of $3 \times 1 0 ^ { - 5 }$ and an effective batch size of 64. The reported training time includes online rollout generation, reliability judgment, rewriting, and verification.

For rollout and image generation, we use 50 diffusion steps with global normalization. The text and image classifier-free guidance scales are set to 4.0 and 1.0, respectively. Each prompt is assigned a fixed random seed shared across compared models.

## A.2 OPENING ARENA PROTOCOL

We follow the official pairwise arena protocol of OpenING (Zhou et al., 2025b). The evaluation set contains 2,160 samples. For each sample, two distinct entries are paired through circular rotation and random permutation, and their presentation order is randomized to mitigate position bias.

The compact comparison in the main paper contains eight entries and 4,320 pairwise battles, with each entry participating in 1,080 battles on average. The full leaderboard in Table 8 contains 15 entries, including our Prism-trained models, their corresponding base models, and 11 official Open-ING reference outputs. It contains 4,143 valid pairwise battles. The number of battles per entry varies between 507 and 615, as reported in the final column of Table 8. Since the compact comparison and the full leaderboard are evaluated in separate arenas, their win rates are arena-relative and are not directly comparable.

We use the official IntJudge-8B checkpoint and decoding configuration. For each battle, IntJudge outputs one of four decisions: A, B, Tie(A), or Tie(B). We report four tie-handling metrics:

• FDT: Tie(A) is counted as a win for A, and Tie(B) is counted as a win for B.

• w/o Tie: only battles with an explicit winner are counted.

• w/ Tie (0): ties contribute zero wins to both models.

• w/ Tie (.5): ties contribute 0.5 wins to both models.

We use w/ Tie (.5) as the main reporting protocol. The Human entry uses the human-authored interleaved image–text sequences released with OpenING and participates in the same arena protocol as model outputs. Under the w/o Tie protocol, SenseNova-U1 with Prism achieves an 86.61% win rate over 478 decisive outcomes from 537 battles, exceeding the base model by 3.55 points and GPT-4o+DALL-E3 by 5.23 points.

Table 8: Full OpenING arena leaderboard. We report pairwise win rates under four tie-handling protocols. The final column reports decisive battles over total battles for each entry.
<table><tr><td>Rank</td><td>Model</td><td>FDT</td><td>w/o Tie</td><td>w/ Tie (0)</td><td>w/ Tie (.5)</td><td>Decisive / Total</td></tr><tr><td>1</td><td>SenseNova-U1 + Prism</td><td>84.73%</td><td>86.61%</td><td>77.09%</td><td>82.59%</td><td>478 / 537</td></tr><tr><td>2</td><td>SenseNova-U1</td><td>82.33%</td><td>83.06%</td><td>71.91%</td><td>78.62%</td><td>490 / 566</td></tr><tr><td>3</td><td>GPT-4o+DALL-E3</td><td>77.78%</td><td>81.38%</td><td>69.79%</td><td>76.91%</td><td>494 / 576</td></tr><tr><td>4</td><td>Human</td><td>76.88%</td><td>82.92%</td><td>67.50%</td><td>76.80%</td><td>486 / 597</td></tr><tr><td>5</td><td>Gemini1.5+Flux</td><td>62.19%</td><td>64.16%</td><td>50.36%</td><td>61.11%</td><td>438 / 558</td></tr><tr><td>6</td><td>BAGEL + Prism</td><td>54.38%</td><td>54.79%</td><td>42.93%</td><td>53.76%</td><td>438 / 559</td></tr><tr><td>7</td><td>SEED-LLaMA</td><td>43.32%</td><td>47.04%</td><td>35.92%</td><td>47.74%</td><td>423 / 554</td></tr><tr><td>8</td><td>Anole</td><td>48.66%</td><td>44.18%</td><td>35.50%</td><td>45.32%</td><td>421 /524</td></tr><tr><td>9</td><td>BAGEL-7B-MoT</td><td>43.76%</td><td>40.89%</td><td>33.59%</td><td>42.51%</td><td>428 / 521</td></tr><tr><td>10</td><td>Emu3</td><td>42.59%</td><td>37.56%</td><td>30.99%</td><td>39.73%</td><td>434 / 526</td></tr><tr><td>11</td><td>Emu2</td><td>36.32%</td><td>36.20%</td><td>27.54%</td><td>39.50%</td><td>442 / 581</td></tr><tr><td>12</td><td>MiniGPT-5</td><td>26.02%</td><td>27.08%</td><td>21.14%</td><td>32.11%</td><td>480 / 615</td></tr><tr><td>13</td><td>SEED-X</td><td>30.84%</td><td>29.17%</td><td>27.48%</td><td>30.37%</td><td>504 / 535</td></tr><tr><td>14</td><td>MiniGPT-5mmd</td><td>20.57%</td><td>17.36%</td><td>14.91%</td><td>21.98%</td><td>455 / 530</td></tr><tr><td>15</td><td>NExT-GPT</td><td>14.20%</td><td>10.33%</td><td>9.27%</td><td>14.40%</td><td>455 /507</td></tr></table>

Table 9: OneIG-Bench results. Each model generates four images per prompt on 1,120 English prompts. The five dimensions are scored with dedicated scorers and the benchmark provides no aggregate score. The better value is in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Prism</td><td colspan="5">OneIG-Bench</td></tr><tr><td>Alignment</td><td>Text</td><td>Style</td><td>Reasoning</td><td>Diversity</td></tr><tr><td rowspan="2">SenseNova-U1</td><td>x</td><td>0.873</td><td>0.942</td><td>0.405</td><td>0.326</td><td>0.170</td></tr><tr><td>√</td><td>0.869</td><td>0.937</td><td>0.410</td><td>0.330</td><td>0.183</td></tr><tr><td rowspan="2">BAGEL</td><td>x</td><td>0.813</td><td>0.366</td><td>0.395</td><td>0.189</td><td>0.176</td></tr><tr><td>√</td><td>0.808</td><td>0.339</td><td>0.395</td><td>0.189</td><td>0.179</td></tr></table>

## A.3 IMAGE GENERATION EVALUATION

We evaluate image generation with the official data and scoring pipelines of GenEval (Ghosh et al., 2023) and DPG-Bench (Hu et al., 2024). GenEval contains 553 prompts, with four images generated per prompt. It evaluates six categories using Mask2Former and CLIP, and reports their equal-weighted average. DPG-Bench contains 1,065 dense prompts and uses mPLUG-VQA for evaluation.

All images are generated at 1024 × 1024 using the configuration above. For DPG-Bench, we generate four images per prompt and arrange them into a 2 × 2 grid. The official evaluator crops and scores each image at the native 1024 resolution. All comparisons with the base models use identical generation settings.

## A.4 MULTIMODAL UNDERSTANDING EVALUATION

We evaluate multimodal understanding using the official data, prompts, and evaluation scripts of MME, MMBench, MMMU, MMStar, POPE, and MMVP. No task-specific prompts are added.

MME contains 14 tasks, including 10 perception tasks with a maximum score of 2,000 and four cognition tasks with a maximum score of 800. We report the total score in the main paper. MMBench is evaluated on its 1,176-example development split with the official option-rotation protocol. MMMU is evaluated on its 900-example validation split, including 847 multiple-choice and 53 open-ended questions. We report multiple-choice accuracy. MMStar contains 1,500 examples. POPE includes 9,000 questions from the random, popular, and adversarial settings, for which we report Accuracy and F1. MMVP contains 300 questions organized into 150 paired examples, and is scored only when both questions in a pair are answered correctly.

All understanding benchmarks are evaluated using a unified decoding configuration and the official aggregation scripts.

## B ADDITIONAL CAPABILITY ANALYSIS

Prism constructs supervision from the consistency between the complete text and image chains of self-generated interleaved trajectories. Its training objectives therefore emphasize planning, visual state progression, and cross-modal consistency, rather than instruction following for single-image generation or image editing. This appendix reports results on benchmarks that fall outside this scope. All benchmarks are evaluated with the official protocols and judge models released by the benchmark authors.

OneIG-Bench. OneIG-Bench evaluates text-to-image generation on 1,120 English prompts with four images generated per prompt (Chang et al., 2026). As shown in Table 9, the results are mixed but close to the base models. On SenseNova-U1, Prism improves Diversity by 0.013, Style by 0.005, and Reasoning by 0.004, while Alignment and Text decrease slightly by about 0.005. The improved dimensions are consistent with training on trajectories that require coordinated textual plans and visual states. On BAGEL, Style, Reasoning, and Diversity are unchanged, Alignment decreases slightly, and Text shows the largest decrease of 0.027. Text rendering is the weakest dimension of BAGEL among the five, with 0.366 compared with 0.942 on SenseNova-U1, and it remains the weakest after training. In summary, Prism preserves most OneIG capabilities, improves the dimensions aligned with its training objectives, and causes small decreases on Alignment and Text, which lie outside these objectives.

Table 10: TIIF-Bench and editing results. TIIF-Bench uses a short and a long prompt variant per prompt with Qwen2.5-VL-72B as the judge. ImgEdit-Bench scores 1–5 and GEdit-Bench (English) uses VIEScore on a 10-point scale. The better value is in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Prism</td><td colspan="2">TIIF-Bench</td><td colspan="3">Editing</td></tr><tr><td>Short</td><td>Long</td><td>GEdit-S</td><td>GEdit-Q</td><td>ImgEdit</td></tr><tr><td rowspan="2">SenseNova-U1</td><td>x</td><td>0.807</td><td>0.805</td><td>7.37</td><td>6.79</td><td>3.58</td></tr><tr><td>√</td><td>0.812</td><td>0.794</td><td>7.23</td><td>6.52</td><td>3.47</td></tr><tr><td rowspan="2">BAGEL</td><td>x</td><td>0.699</td><td>0.699</td><td>6.89</td><td>6.93</td><td>3.33</td></tr><tr><td>√</td><td>0.664</td><td>0.681</td><td>6.82</td><td>6.87</td><td>3.38</td></tr></table>

TIIF-Bench. TIIF-Bench evaluates instruction following under short and long prompt variants, scored by Qwen2.5-VL-72B over eleven sub-attributes (Wei et al., 2025). As shown in Table 10, on SenseNova-U1 Prism slightly increases the short score and slightly decreases the long score, while on BAGEL both scores decrease. On SenseNova-U1, the basic sub-attributes improve under both variants, the advanced ones improve under short prompts and decrease under long ones, and style control decreases under both.

These patterns follow from how the training organizes generation. Every image in a trajectory is generated with the preceding text and images as context and realizes only the incremental state of the current step, so the training reinforces continuation from rich context. A short prompt is close to realizing a single simple state and benefits. A long prompt instead requires satisfying all constraints in one generation from text alone, and style control requires the same one-shot fidelity, so neither is reinforced and both decrease. On BAGEL, which starts weaker on single-image instruction following, the gains do not materialize and the decreases spread across nearly all sub-attributes.

Editing. ImgEdit-Bench and GEdit-Bench evaluate instruction-based editing, scored by ImgEdit-Judge and VIEScore respectively (Ye et al., 2026; Liu et al., 2025). As shown in Table 10, editing results are mixed. On SenseNova-U1 both benchmarks decrease slightly, with the largest per-task decrease on style change. On BAGEL, ImgEdit-Bench slightly increases, and the increases concentrate on content-oriented operations such as extraction and person editing, while GEdit-Bench changes by less than 0.07.

The transfer is limited because the two tasks use the conditioning image differently. In a trajectory, the preceding images serve as state history, and each new image re-renders the updated state, inheriting entities and progress semantically while sharing few pixels with its predecessors. Instructionbased editing instead keeps most pixels of the input image and changes only the specified region. The training therefore reinforces semantic continuation conditioned on images but never pixel-level preservation, which explains why content-oriented operations improve while tasks that demand pixel fidelity stay flat or decrease. Obtaining editing gains requires editing-specific supervision as in editing-oriented methods such as Uni-Edit (Zheng et al., 2026).

## C DETAILED RELIABILITY JUDGMENT AND ROUTING ANALYSIS

We further examine the reliability judgments and training targets underlying the results in Table 6. All variants start from 2,395 prompt seeds and use the same one-epoch optimization budget and learning rate. Base w/ Self-Training optimizes unmodified self-generated rollouts with L<sub>interleave</sub>. Prism w/ Random Routing preserves the number of trajectories assigned to each action within a minibatch but permutes their assignments independently of the judged reliability states. Prism w/o Rewriting retains reliability-based routing but trains trajectories assigned to t2i and i2t on their unmodified rollouts instead of constructing rewritten targets.

Full Prism outperforms Base w/ Self-Training and Prism w/ Random Routing on all seven metrics in Table 6, and Prism w/o Rewriting on six of seven. Relative to Random Routing, it improves GenEval from 87.6 to 89.2, MME from 2297.5 to 2311.6, and MMVP from 59.3 to 62.0. These comparisons support the value of matching training actions to judged reliability states and constructing rewritten targets, beyond direct self-training or preserving the overall action mixture alone. We next examine the frequency of each reliability state, the evolution of routing during training, and the verification of rewritten targets. Tables 11 and 13 describe the first rollout of each prompt seed, whereas Table 12 describes all rollouts judged during the training epoch.

## C.1 EMPIRICAL ROUTING DISTRIBUTION

Table 11 reports the routing decisions for the first rollout of each of the 2,395 SenseNova-U1 prompt seeds. Both chains pass in 52.2% of these rollouts, while both fail in 14.7%. The remaining 794 rollouts (33.2%) have one reliable and one unreliable chain: 571 (23.8%) are assigned to t2i and 223 (9.3%) to i2t. Thus, asymmetric reliability occurs in both directions. Trajectory-specific routing allows Prism to condition each correction on the chain judged reliable, rather than applying a fixed rewriting direction to all asymmetric cases.

Table 11: Routing decisions for the first rollout of each of the 2,395 SenseNova-U1 prompt seeds.
<table><tr><td>Action</td><td>Reliability state</td><td>Count</td><td>Share</td></tr><tr><td>Pass</td><td>both chains reliable</td><td>1,250</td><td>52.2%</td></tr><tr><td>t2i</td><td>text reliable, image flawed</td><td>571</td><td>23.8%</td></tr><tr><td>i2t</td><td>text flawed, image reliable</td><td>223</td><td>9.3%</td></tr><tr><td>Fail</td><td>both chains flawed</td><td>351</td><td>14.7%</td></tr></table>

## C.2 ROUTING DYNAMICS DURING TRAINING

Table 12 tracks routing across the 38 updates of the one-epoch Full Prism run. Unlike Table 11, it includes all rollouts judged during training. When a rewritten candidate fails verification, that candidate is discarded and a replacement rollout is generated from a new prompt. The replacement is judged anew and may be assigned to any of the four actions. Consequently, the stage-wise percentages characterize the online training stream rather than the fixed first-rollout cohort. The judge remains a frozen copy of the base model throughout.

The routing distribution changes over the epoch. From S1 to S5, the Pass share rises from 49.3% to 59.3%, while the Fail share falls from 17.9% to 9.7%. The i2t share declines from 11.5% to 6.8%; in contrast, t2i rises initially and remains near one quarter of judged rollouts in later stages. The text-chain pass rate increases by 12.9 points, from 70.6% to 83.5%, whereas the image-chain pass rate increases by 5.3 points, from 60.8% to 66.1%. With the judge held fixed, these trends are consistent with improving rollout reliability, particularly on the text side, while image-side failures remain a substantial source of correction targets. Because failed candidates are replaced with newly sampled prompts, the stage-wise changes also reflect the composition of the online rollout stream.

Table 12: Routing dynamics across the 38 updates of one epoch on SenseNova-U1. Statistics include all judged rollouts, including replacements generated after failed rewrite verification. The judge is frozen throughout. Text and image pass rates are computed from the corresponding routing categories. All values are percentages.
<table><tr><td rowspan="2">Stage</td><td rowspan="2">Updates</td><td colspan="4">Routing Distribution</td><td colspan="2">Judge Pass Rate</td></tr><tr><td>Pass</td><td>t2i</td><td>i2t</td><td>Fail</td><td>Text</td><td>Image</td></tr><tr><td>S1</td><td>0-7</td><td>49.3%</td><td>21.3%</td><td>11.5%</td><td>17.9%</td><td>70.6%</td><td>60.8%</td></tr><tr><td>S2</td><td>8-15</td><td>51.6%</td><td>25.1%</td><td>8.7%</td><td>14.6%</td><td>76.7%</td><td>60.3%</td></tr><tr><td>S3</td><td>16-23</td><td>55.7%</td><td>25.2%</td><td>7.7%</td><td>11.4%</td><td>80.9%</td><td>63.4%</td></tr><tr><td>S4</td><td>24-30</td><td>57.6%</td><td>24.7%</td><td>7.1%</td><td>10.6%</td><td>82.3%</td><td>64.7%</td></tr><tr><td>S5</td><td>31-37</td><td>59.3%</td><td>24.2%</td><td>6.8%</td><td>9.7%</td><td>83.5%</td><td>66.1%</td></tr></table>

## C.3 VERIFICATION OF REWRITTEN TARGETS

A trajectory assigned to t2i or i2t is used as a positive rewritten target only if its correction passes verification. Table 13 reports verification outcomes for the rewriting candidates arising from the first-rollout cohort in Table 11. Of the 571 t2i candidates, 376 (65.8%) are accepted and 195 (34.2%) are discarded. Of the 223 i2t candidates, 117 (52.5%) are accepted and 106 (47.5%) are discarded. The lower acceptance rate for i2t shows that the two directions have different verification outcomes;

Table 13: Verification outcomes for rewriting candidates arising from the first-rollout cohort in Table 11. Accepted and discarded counts are candidate-level terminal outcomes; verification calls include retries for these candidates.
<table><tr><td>Rewrite</td><td>Candidates</td><td>Accepted</td><td>Discarded</td><td>Verification Calls</td></tr><tr><td>t2i</td><td>571</td><td>376 (65.8%)</td><td>195 (34.2%)</td><td> $1 { , } 4 7 4 \left( I ^ { + } \mathrm { \ r e v i e w s } \right)$ </td></tr><tr><td>i2t</td><td>223</td><td>117 (52.5%)</td><td>106 (47.5%)</td><td> $3 2 0 ( T ^ { + } \mathrm { r e v i e w s } )$ </td></tr></table>

neither set of generated corrections is admitted wholesale. Discarded candidates do not become positive training targets and are replaced with newly generated rollouts, which are judged and routed independently.

Verification also entails different amounts of review in the two directions. The first-rollout t2i candidates require $1 { , } 4 7 4 I ^ { + }$ review calls, compared with $3 2 0 T ^ { + }$ review calls for i2t candidates. These counts include retries, rather than additional training trajectories. Together, the acceptance rates and review counts characterize verification as a target-selection step: routing identifies a chain to correct, while verification determines whether the resulting correction can be used for training.

## C.4 JUDGE RELIABILITY, CONTAMINATION, AND EXTERNAL-JUDGE COMPARISON

Routing and verification depend on judgments that may themselves be incorrect. We therefore compare the frozen SenseNova-U1 judge used by Full Prism with Qwen3.7 Plus in a separate judge comparison. We sample 200 training rollouts and obtain human annotations for their complete text and image chains, yielding 400 chain-level judgments. A positive judge prediction denotes a chain judged reliable. Table 14 reports agreement with the human annotations and the false discovery rate (FDR), defined as $\mathrm { F P / ( T P } ^ { \bullet } + \mathrm { F P } )$ . FDR measures the fraction of chains accepted as reliable that are flawed according to the annotations, and is therefore directly relevant to the quality of accepted teacher signals.

Table 14: Chain-level agreement with human annotations on 200 rollout trajectories. Each text or image row contains 200 judgments; the All rows aggregate 400 judgments. FDR is $\mathrm { F P } / ( \mathrm { T P } + \mathrm { F P } )$
<table><tr><td rowspan="2">Judge</td><td rowspan="2">Chain</td><td colspan="4">Human Agreement</td><td colspan="3">Judge Quality</td></tr><tr><td>TP</td><td>FP</td><td>TN</td><td>FN</td><td>Acc.</td><td>F1</td><td>FDR</td></tr><tr><td rowspan="3">SenseNova-U1 (frozen)</td><td>Text</td><td>113</td><td>18</td><td>47</td><td>22</td><td>80.0%</td><td>85.0%</td><td>13.7%</td></tr><tr><td>Image</td><td>95</td><td>17</td><td>64</td><td>24</td><td>79.5%</td><td>82.3%</td><td>15.2%</td></tr><tr><td>All</td><td>208</td><td>35</td><td>111</td><td>46</td><td>79.8%</td><td>83.7%</td><td>14.4%</td></tr><tr><td rowspan="3">Qwen3.7 Plus</td><td>Text</td><td>128</td><td>24</td><td>41</td><td>7</td><td>84.5%</td><td>89.2%</td><td>15.8%</td></tr><tr><td>Image</td><td>87</td><td>7</td><td>74</td><td>32</td><td>80.5%</td><td>81.7%</td><td>7.4%</td></tr><tr><td>All</td><td>215</td><td>31</td><td>115</td><td>39</td><td>82.5%</td><td>86.0%</td><td>12.6%</td></tr></table>

Qwen3.7 Plus has higher pooled accuracy and F1 than the frozen judge: 82.5% versus 79.8% and 86.0% versus 83.7%, respectively. However, its advantage does not extend uniformly to the chains accepted as reliable. The frozen judge has a lower text-chain FDR (13.7% versus 15.8%), while Qwen3.7 Plus has a lower image-chain FDR (7.4% versus 15.2%). Thus, higher aggregate agreement with human annotations does not imply lower contamination among accepted signals in both modalities.

The consequences ofjudgment errors also depend on their type. A false negative may reject a usable teacher chain; if the other chain is accepted, it can additionally trigger an unnecessary rewrite. When both chains are rejected, an otherwise reliable text chain may receive the unlikelihood objective. Fail accounts for 14.7% of the first rollouts in Table 11, although this share does not measure the frequency of such erroneous assignments. A false positive can instead admit a flawed chain as a teacher signal. For rewritten targets, verification provides an additional gate: among the first-rollout candidates in Table 13, it discards 34.2% of t2i and 47.5% of i2t candidates. These measurements establish neither error-free routing nor error-free accepted targets, but identify where judgment noise enters the training pipeline and where verification filters its outputs.

The i2t results connect this analysis to the loss ablation. Replacing $\mathcal { L } _ { \mathrm { i 2 t } }$ with $\mathcal { L } _ { \mathrm { i n t e r l e a v e } }$ on the corresponding unmodified rollouts lowers MMVP from 62.0 to 56.7 (Table 5). Meanwhile, verification accepts 117 of the 223 first-rollout i2t candidates and rejects 106 (Table 13). The ablation supports the contribution of the i2t training objective relative to direct training on the same rollouts, while the verification results show that its rewritten targets are filtered before use.

We also compare the downstream results obtained with the two judges. As shown in Table 15, the external judge yields slightly higher MMBench and MMMU scores, whereas the frozen SenseNova-U1 judge yields higher MME, POPE F1, MMVP, GenEval, and DPG; the two tie on MMStar. Figure 8 illustrates a corresponding difference in judgment emphasis: Qwen3.7 Plus identifies a local omission in the text chain, while SenseNova-U1 identifies an image-count violation in the complete trajectory. The external judge therefore changes the resulting capability profile rather than improving it uniformly. Together with the comparison against Random Routing in Table 6, these results support the use of a frozen internal judge as an informative, albeit imperfect, routing signal.

Table 15: Effects of internal and external judges on SenseNova-U1. POPE reports F1.
<table><tr><td>Judge</td><td>MME</td><td>MMBench</td><td>MMMU MC</td><td>MMStar</td><td>POPE</td><td>MMVP</td><td>GenEval</td><td>DPG</td></tr><tr><td>Base</td><td>2282.6</td><td>86.1</td><td>59.0</td><td>65.1</td><td>84.0</td><td>58.0</td><td>85.8</td><td>86.1</td></tr><tr><td>SenseNova-U1</td><td>2311.6</td><td>87.3</td><td>60.6</td><td>66.5</td><td>86.6</td><td>62.0</td><td>89.2</td><td>87.3</td></tr><tr><td>Qwen3.7 Plus</td><td>2304.1</td><td>87.4</td><td>60.7</td><td>66.5</td><td>86.3</td><td>59.3</td><td>88.9</td><td>86.9</td></tr></table>

## D THE IMPACT OF FREEZING STRATEGY

We examine the role of freezing the judge during online training by comparing Full Prism with a variant in which the judge is updated jointly with the rollout model. The comparison uses the same training setup and tracks two quantities. First, the routing distribution records how the judge assigns generated trajectories to the four actions; the text- and image-chain pass rates are computed from these assignments. Second, the zero-image rate is the proportion of generated interleaved trajectories containing no image output and serves as a measure of image-generation collapse. The jointly updated run terminates after 34 updates, whereas the frozen-judge run completes all 38 updates.

Table 16 shows that the two settings have similar routing distributions in the first stage but subsequently diverge. With the frozen judge, the text-chain pass rate increases from 70.6% to 83.5% and the image-chain pass rate from 60.8% to 66.1%, while all four actions remain active throughout the epoch. The Frozen judge rows reproduce the online training statistics in Table 12. With joint updates, the Pass share rises from 50.6% in the first stage to 87.4% in updates 16–23, while i2t and Fail fall to 4.8% and 1.0%. The increasing concentration in Pass indicates that joint updates substantially change the routing signal.

This concentration does not persist as stable generation. In the final two updates of the jointly updated run, the Pass share falls to 46.0% and t2i rises to 54.0%, alongside a drop in the imagechain pass rate to 46.0%. Table 17 shows that its final zero-image rate reaches 70.59%, compared with 0.00% for the frozen-judge run. Thus, the high intermediate pass rates of the jointly updated judge do not correspond to sustained image-generation quality. In this comparison, freezing the judge maintains a stable routing reference and avoids the observed collapse.

## E IMPLEMENTATION OF TRAINING OBJECTIVES

Equation 3 defines the objective for one interleaved trajectory $\tau = ( T _ { 1 } , I _ { 1 } , \dots , T _ { K } , I _ { K } )$ , where $P$ is the task prompt, $T _ { i }$ is the i-th text segment, and $I _ { i }$ is the image generated after $T _ { i }$ . Each step follows a text-first, then image order. The trajectory-level objective sums a text loss and an image loss over the K interleaved steps.

For the text side, let $T _ { i } = ( t _ { i , 1 } , \dots , t _ { i , | T _ { i } | } )$ denote the token sequence of the i-th text segment. We use standard token-level next-token cross-entropy:

$$
\mathcal { L } _ { \mathrm { u n d } } ^ { ( i ) } = - \sum _ { n = 1 } ^ { | T _ { i } | } \log p _ { \theta } ( t _ { i , n } \mid P , T _ { < i } , I _ { < i } , t _ { i , < n } ) ,\tag{10}
$$

where $T _ { < i } = ( T _ { 1 } , \dots , T _ { i - 1 } )$ is the sequence of preceding text segments, $I _ { < i } = ( I _ { 1 } , \ldots , I _ { i - 1 } )$ is the sequence of preceding images, and $t _ { i , < n } = ( t _ { i , 1 } , \ldots , t _ { i , n - 1 } )$ are the previous tokens in the current text segment.

## Task

A farm changed the color of its weather vanes. The rooster weather vane was RED, and the farmer painted it BLACK. The arrow weather vane was GOLD, and he painted it SILVER. Draw the barn roof before and after. For each drawing, say in words the color of each weather vane.

The farm's barn roof is covered in weathered, dark brown wooden shingles, with a few patches of green moss growing in the crevices. At the very peak of the roof sits a large, ornate rooster weather vane. It is painted a brilliant, glossy red, its tail feathers fanned out and its comb a deep crimson. The sky behind it is a clear, pale blue, making the red rooster stand out sharply against the rustic wood. The scene is peaceful and bright, capturing the traditional look of the farm before the farmer's changes.   
<image>

![](images/d1b74fcdd29e3258e6e1f0deb5c6a50813a90b39154488ec659fb5b8a66d0e38.jpg)  
Qwen3.7-Plus

The farmer has now finished his work, and the scene has shifted. The once-vibrant red rooster weather vane is now a deep, matte black. The paint is thick and opaque, giving the rooster a more somber and mysterious appearance. The dark shingles of the roof and the clear blue sky remain the same, but the striking contrast of the black rooster against the light background creates a much more dramatic and moody atmosphere. The transformation is complete, with the weather vane now serving as a dark silhouette against the horizon. <image>

Moving to a different part of the roof, we see a classic arrow weather vane. It is made of polished metal and painted a brilliant, reflective gold. The arrow points toward the right, its fletching and tip gleaming in the sunlight. The golden color is rich and warm, catching the light and casting a soft glow on the surrounding wooden shingles. The sky is a soft, hazy blue, providing a gentle backdrop that allows the golden arrow to shine as a symbol of direction and prosperity.   
<image>

The farmer has also repainted the arrow weather vane. It is now a cool, polished silver. The metallic surface is highly reflective, mirroring the pale blue of the sky and the dark tones of the roof. The silver paint gives the arrow a sleek, modern, and industrial feel, contrasting with the rustic, weathered wood of the barn. The arrow still points to the right, but its new color gives it a colder, more clinical appearance, standing out as a bright, metallic focal point against the textured shingles. <image>

In this final view, we see the entire roofline of the barn. Both the black rooster and the silver arrow weather vanes are visible, standing side-by-side at the peak. The black rooster is on the left, its dark form providing a strong anchor, while the silver arrow on the right   
catches the light with its metallic sheen. The dark brown shingles and the clear blue sky frame the scene, showing the completed transformation of the farm's symbols. The two weather vanes now stand together, representing the new, updated look of the barn.   
<image>

![](images/c06569fe1b2af13f2c10b00a9a0d4ed9bda1ff90dc055ad9e03d65f2cd739b5a.jpg)

![](images/e8bd7700fcadaa51bc7c970f67309a8fc98c8a6a992a8721eb7bc8b14be0ec34.jpg)

![](images/128a2caa1f8a47ffcd3ecdccd3bd61e921020fa02b62e7c89d67a7cdc801fe4f.jpg)

![](images/b1860642fc7440ae0b148fe3ac25bcd291df403335cb93467ed3dcf512ea2bb9.jpg)

![](images/fa20e008b61827158a8222e37f97e51c7cbe1ed9ac85072baf4fcb74b99f8463.jpg)  
Figure 8: Different judgment emphases of Qwen3.7 Plus and SenseNova-U1. Qwen3.7 Plus identifies a local omission, while SenseNova-U1 identifies an image-count violation in the complete trajectory.

## Text Chains review

The task requires stating the color of each weather vane for each drawing, but the first drawing only describes the rooster and omits the arrow weather vane.

![](images/0fcae6542530be5158c624cfac54b22c2221c8bb573512a1ffaa5654beddc6de.jpg)  
Base

The task requires drawing the barn roof before and after, implying exactly two images total. The solution text includes five <image> markers, violating the explicit image-count requiremen t for this specific task.

## Image Chains review

![](images/aa49add6a75546257859209d977d1bccc16f1b342a3b2caef6faa3379cd3819a.jpg)  
Qwen3.7-Plus

The task requires stating the color of each weather vane for each drawing, but the first drawing only describes the rooster and omits the arrow weather vane.

![](images/205de820679c786395337e62bd02a77f8efd29a405bc5b339b7a47e43a2cef33.jpg)

The task requires drawing the barn roof before and after, implying exactly two images total. The solution text includes five <image> markers, violating the e x p l i c i t i m a g e - c o u n t r e q u i r e m e n t f o r t h i s specific task.

For the image side, let $z _ { i , 1 }$ denote the clean target visual representation of image $I _ { i }$ under the native image representation of the backbone. This may be a latent code, patch representation, or another backbone-specific continuous representation. Let $z _ { i , 0 } \sim p _ { 0 }$ denote a noise sample of the same shape drawn from a fixed noise distribution. We sample an interpolation time $s \sim \mathcal { U } [ 0 , 1 ]$ and define the linear flow-matching path

$$
z _ { i , s } = ( 1 - s ) z _ { i , 0 } + s z _ { i , 1 } ,\tag{11}
$$

with target velocity

$$
v _ { i } ^ { \star } = z _ { i , 1 } - z _ { i , 0 } .\tag{12}
$$

Let

$$
c _ { i } = ( P , T _ { \leq i } , I _ { < i } ) , \qquad T _ { \leq i } = ( T _ { 1 } , \dots , T _ { i } ) ,\tag{13}
$$

denote the conditioning context for generating $I _ { i } .$ . The model predicts a velocity field $v _ { \theta } ( z _ { i , s } , s , c _ { i } )$ and the image-generation loss is

$$
\mathcal { L } _ { \mathrm { g e n } } ^ { ( i ) } = \mathbb { E } _ { s \sim \mathcal { U } [ 0 , 1 ] , z _ { i , 0 } \sim p _ { 0 } } \left[ \left\| v _ { \theta } ( z _ { i , s } , s , c _ { i } ) - v _ { i } ^ { \star } \right\| _ { 2 } ^ { 2 } \right] .\tag{14}
$$

For the both-pass case, both modalities are judged reliable. We keep the original image chain $I =$ $( I _ { 1 } , \ldots , I _ { K } )$ and reconstruct the full concatenated text chain $T = ( t _ { 1 } , \cdot \cdot \cdot , t _ { | T | } )$ , where $\bar { T }$ is obtained

Table 16: Routing distributions and judge pass rates across training stages under frozen and jointly updated judges. Text and image pass rates are computed from the corresponding routing categories. All values are percentages. The jointly updated run terminates after 34 updates.
<table><tr><td rowspan="2">Judge Setting</td><td rowspan="2">Updates</td><td colspan="4">Routing Distribution</td><td colspan="2">Judge Pass Rate</td></tr><tr><td>Pass</td><td>t2i</td><td>i2t</td><td>Fail</td><td>Text</td><td>Image</td></tr><tr><td rowspan="5">Frozen judge (Ours)</td><td>0-7</td><td>49.3</td><td>21.3</td><td>11.5</td><td>17.9</td><td>70.6</td><td>60.8</td></tr><tr><td>8-15</td><td>51.6</td><td>25.1</td><td>8.7</td><td>14.6</td><td>76.7</td><td>60.3</td></tr><tr><td>16-23</td><td>55.7</td><td>25.2</td><td>7.7</td><td>11.4</td><td>80.9</td><td>63.4</td></tr><tr><td>24-30</td><td>57.6</td><td>24.7</td><td>7.1</td><td>10.6</td><td>82.3</td><td>64.7</td></tr><tr><td>31-37</td><td>59.3</td><td>24.2</td><td>6.8</td><td>9.7</td><td>83.5</td><td>66.1</td></tr><tr><td rowspan="5">Jointly updated judge</td><td>0-7</td><td>50.6</td><td>20.8</td><td>13.2</td><td>15.4</td><td>71.4</td><td>63.8</td></tr><tr><td>8-15</td><td>76.0</td><td>14.4</td><td>4.8</td><td>4.8</td><td>90.4</td><td>80.8</td></tr><tr><td>16-23</td><td>87.4</td><td>6.9</td><td>4.8</td><td>1.0</td><td>94.3</td><td>92.2</td></tr><tr><td>24-31</td><td>82.2</td><td>17.1</td><td>0.5</td><td>0.2</td><td>99.3</td><td>82.7</td></tr><tr><td>32-33</td><td>46.0</td><td>54.0</td><td>0.0</td><td>0.0</td><td>100.0</td><td>46.0</td></tr></table>

Table 17: Effect of freezing the judge on image-generation stability during online training. Lower zero-image rates are better.
<table><tr><td>Judge Setting</td><td>Initial Zero-Image Rate</td><td>Final Zero-Image Rate</td></tr><tr><td>Frozen judge (Ours)</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Jointly updated judge</td><td>0.00%</td><td>70.59%</td></tr></table>

by concatenating all text segments $T _ { 1 } , \dots , T _ { K }$ . The recovery instruction $P _ { \mathrm { r e c } }$ and image chain jointly condition the reconstruction:

$$
\mathcal { L } _ { \mathrm { p a s s } } = - \sum _ { n = 1 } ^ { | T | } \log p _ { \theta } ( t _ { n } \mid P _ { \mathrm { r e c } } , I , t _ { < n } ) .\tag{15}
$$

For the t2i and i2t cases, one modality is corrected and the resulting rewritten trajectory is verified before training. We then apply Equation 3 to the verified corrected trajectory. Thus, every text token is trained with the text loss and every image representation is trained with the image loss.

For the both-fail case, the original trajectory τ is treated as unreliable and is not used as a positive target. Instead, we apply token-level unlikelihood training to the original text chain:

$$
\mathcal { L } _ { \mathrm { f a i l } } ( \tau ) = - \sum _ { i = 1 } ^ { K } \sum _ { n = 1 } ^ { | T _ { i } | } \log \left( 1 - p _ { \theta } \big ( t _ { i , n } ~ | ~ P , T _ { < i } , I _ { < i } , t _ { i , < n } \big ) \right) .\tag{16}
$$

In implementation, the argument of the logarithm is numerically clipped away from zero for stability. Each trajectory $\tau$ is assigned to exactly one reliability case, pass, t2i, i2t, or fail. During minibatch training, only the objective associated with the assigned case contributes to the loss for that trajectory, while the remaining case-specific objectives are masked out. Eq. 9 denotes the resulting aggregate minibatch objective.

## F VERIFICATION OF REWRITTEN TRAJECTORIES

For the t2i and i2t cases, Prism first generates a corrected chain from the preserved reliable chain and the detected error. Text-to-image rewriting generates $I ^ { + }$ from $P , T ,$ , and $r _ { \mathrm { i m g } } ,$ while image-to-text rewriting generates $T ^ { + }$ from $\bar { P , I , }$ and $r _ { \mathrm { t x t } }$ . Each rewritten candidate is verified before it is used as a positive training target. The verifier is a frozen copy of the base model. $\mathbf { A }$ candidate is regenerated for at most four attempts, including the initial generation and three retries. Candidates that fail all attempts are discarded.

Image-side verification. For a rewritten image chain $I ^ { + }$ generated from the preserved text chain $T ,$ the verifier receives the task prompt $P$ and the candidate image sequence. A candidate is accepted only if it contains the same number of images as the original trajectory, satisfies the task requirements for objects, counts, attributes, spatial relations, state changes, and step order, and is judged valid by the verifier.

3. The first confirmed visible error determines invalid.

You are a strict reviewer auditing only the image sequence   
of an interleaved solution and its visual state progression.

Step text is not shown. You see {n\_images} images in order.

TASK:   
{prompt}

Decide whether this image chain delivers the progression   
required by the task.

1. Base each decision on visible evidence.

2. To return invalid, cite one concrete image number and its visible error.

4. Determine from the task whether image count is strict or flexible.

5. Check required objects, counts, attributes, spatial relations,   
actions, visual states, lighting, and state progression.

6. A preview or final state is valid only when it does not conflict   
with an explicitly required stage.

7. Do not reject an image chain solely for style drift, texture noise,   
lighting shifts, background changes, or minor visual variation.

Return invalid for a confirmed error such as no task progress,   
wrong image or stage count, wrong quantity, failed state change,   
state mismatch, corrupted key subject, premature final state,   
state regression, identity swap, internally impossible content,   
or failure to convey a required step.

Return exactly one JSON object with a Boolean field named "valid".

Text-side verification. For a rewritten text chain T<sup>+</sup> generated from the preserved image chain I, the verifier receives the task prompt and candidate text. A candidate is accepted only if the verifier returns valid, its interleaved marker structure is valid, its final image reference is valid, and its length is at least half that of the original rollout text.

You are auditing only the visible text of a no-think interleaved   
text-and-image response. Images are audited separately.   
TASK:   
{prompt}   
SOLUTION TEXT:   
{text\_chain}

Judge whether the written content and image marker structure   
satisfy the task.

1. Enforce image-marker count only when the task explicitly specifies it.

3. The text must cover required objects, stages, relations, or scene changes.

5. For exact-one-image tasks, text after the image marker must not   
introduce another scene or visual plan.

6. For tutorials, processes, stories, and lifecycle tasks, check clear   
ordering errors, missing stages, extra stages violating an explicit   
count, or state regression.

Return exactly one JSON object with a Boolean field named "valid".   
When invalid, include a verbatim faulty span in the "quote" field.

## G ADDITIONAL UNDERSTANDING EXAMPLES

Figure 9 presents additional visual question answering examples. Compared with the base model, Prism gives correct responses on diverse tasks involving food-web reasoning, depth perception, chart interpretation, semantic correspondence, motion direction, spatial attributes, legend understanding, and object counting. These examples further illustrate the improved visual grounding and multimodal reasoning observed in the quantitative evaluation.

![](images/8dafadda6ad3ea68dea84164a40e5d5bd9bf0c56a5ea5c3b2eac2f94fb2d86fe.jpg)  
A food web is shown below. Which of the following organisms compete for the deer as a food source?  
A: hawk and snake B: snake and lion C: Mountain lion  
D: none (A) hawk and snake (B) none (C) Mountain lion

![](images/15be249255cfaadd85905e79b0dede178167d2f3927fbc900d606afb12bbded0.jpg)

![](images/cedf234fd86e32f9f2b2ef682b442fda272d61165920248011ee72ca6562b8f3.jpg)

![](images/a05f82c62d09278ca2abc1290b642cbfacb7bfd11e7377a351ce4d9254855678.jpg)  
Two points are circled on the image, labeled by A and B beside each circle. Which point is closer to the camera?  
A: A is closer B: B is closer

![](images/8968c413fb2492ed701860c1a6678e9cc8bc3afbe1117340a575fa8fd6ec99db.jpg)  
Humans can find corresponding points for different objects in the same category. For instance, if there are images of two different cats, then the left ear tip of one cat corresponds to the left ear tip of the other cat, and the right front paw of one cat corresponds to the right front paw of the other cat. Given the following two images, a reference point is annotated on the first image, labeled with REF. You are given multiple red-circled points on the second image, choices of "A, B, C, D" are drawn beside each circle. Select between the choices on the second image and find the corresponding point for the reference point. Which point is corresponding to the reference point?

![](images/22cb96577024a557b2e10f0a6aaa474de305f524070d04af253ac5c43fa8417a.jpg)  
What are the dangers degree represented by yellow, blue, and orange in image?  
A: Vulnerable, Endangered, Critical B: Endangered, Critical, Vulnerable C: Critical, Vulnerable, Endangered  
A: Point A B: Point B C: Point C D: Point D

![](images/566d14d5e8ab6fee997173fb448f06c992be0c37dfc8a207bd60fae9b51b7c25.jpg)

![](images/fc10244785af4a0c5be8a865c5a2db477593b71c561cdc58ba0823d11b61763b.jpg)

![](images/b5b2197efd5b03e95afc672205f444fbacf3b304feede62afa925bf0e5872d36.jpg)

![](images/e001efe0f4a3f98678a0763749895922c202a976903c79a9d4c7708e79731b22.jpg)  
Is the person in the image running towards or away from the camera?  
In this image, are the berries placed  
A: On a dish B: In bowls of yogurts  
A: Towards B: Away from

![](images/e5777396076088929c2fcffef2fe2eea67e97dd8032c533f9e0984a82f2798df.jpg)

![](images/a64e885051b1e88ee0b49ab1980d72163f9018d679432096965f78e355d8eff0.jpg)

![](images/255f419eb5a1f4005db6d025d5126171534d0e38c036b4aa1116c2ef36063b59.jpg)  
How many blue balloons are in the room?  
A: Three B: Two C: One D: Four

![](images/5a348d7db5b1d28c038db52439fc557849bbd3235cf1dd9b37408a55d5de3ad6.jpg)

![](images/bf9468877161d84aba223c3cb0dace70f556c53bcaecb3b448b965321ca110a7.jpg)

![](images/6b67460a64076a89980c2b55a14ab385488969f05be11c005411a9ff10ba180f.jpg)  
Figure 9: Additional VQA examples. Prism corrects errors made by the base model on questions requiring visual reasoning, chart understanding, correspondence matching, spatial perception, and counting.

Figure 10 presents additional generation examples on BAGEL. The examples cover diverse visual styles and contents, including realistic photography, landscape scenes, illustration, portraiture, food, wildlife, and indoor environments.

![](images/b619e1628e9c0c4d77a6d41df9cdf0081761a9d78b9c3b1955cc814de47ae0a0.jpg)  
Figure 10: Additional generation examples on BAGEL. The generated images cover diverse subjects, scenes, and visual styles.

## I ADDITIONAL INTERLEAVED GENERATION EXAMPLES

Figure 11 compares interleaved generation by the base model and Prism. In the picture-book task, the base model repeats similar barn scenes, making the visual progression of the story unclear. For food recommendations, it repeatedly generates a person eating while providing limited visual coverage of the requested dishes. In the solar-eclipse example, the base model produces a long textdominated response with only one illustration, whose cluttered background does not clearly convey the required Earth–Moon–Sun alignment. In contrast, Prism produces distinct visual states for the story, diverse food images for the recommendation, and stepwise text–image explanations of the eclipse.

![](images/9d038867baab9fee791d7e5420ad5d673d559bf4f4060689f303857580cf26ff.jpg)  
Figure 11: Additional interleaved generation examples. Compared with the base model, Prism produces clearer visual progression in a picture-book story, more diverse images for food recommendations, and visual understanding for a solar eclipse.