(b) Parallel thought-image generation.

# PATTS: Parallel-Aware Test-Time Scaling for Thought-Image Generation in Multimodal Discrete Diffusion LLMs

Yasong Dai<sup>1,</sup> <sup>2</sup>, Zeeshan Hayder<sup>1</sup>, David Ahmedt-Aristizabal<sup>2</sup>, Hongdong Li<sup>1,</sup> <sup>3</sup> <sup>1</sup>Australian National University, <sup>2</sup>CSIRO Data61, <sup>3</sup>Amazon

![](images/edef64e4284ffac4f871a1ffb3f72744234aede04d22e8097e8bb1a23a45aa97.jpg)

![](images/2133d446d2859106c39dcc30b692fef553c48b79132c471585c9ad354a4265a7.jpg)

![](images/670e3576e790c00567836f3f8de405ea126a4e12b6bbd8fea444bf2a9b95f311.jpg)  
(a) PATTS trajectory search with reliability-based verification.  
1. Lighting: The current image has shadows ... light, which can enhance the mood and create a more serene atmosphere . 2. Color Tone : The colors in the question image are muted and somewhat desaturated . The answer image has a vibrant blue tint, which could be due to ... . The answer image has a shallow depth of field, focusing on the bench ... and the background . 4. and Texture : The wood in the answer image looks slightly more worn and aged compared to the clean, new bench in the question image ... By applying these adjustments, the image image will better reflect the transformation.  
Figure 1. Overview of PATTS. A parallel diffusion multimodal LLM generates multiple partially decoded thought-image trajectories. At selected cut point, PATTS forms a predicted clean preview $\hat { z } _ { 0 } ^ { ( t ) }$ and combines model-internal reliability diagnostics with an externa Vision-Language-Model (VLM) score. The state-dependent reliability weight c (z ) increases the influence of external semantic verification only when the intermediate preview is sufficiently stable to rank.

## Abstract

Test-time scaling (TTS) improves generative models by allocating additional inference-time computation to sampling, search, and verification. Existing image generation approaches, however, typically rank completed images or image-centric intermediate states. We study TTS for parallel multimodal discrete diffusion language models, in which reasoning tokens and image tokens are decoded jointly within a single generation trajectory. We first diagnose a limitation of conventional final-image selection: on GenEval, a CLIPstyle image-prompt score is weakly aligned with withinprompt compositional correctness, leaving substantial headroom between verifier selection and oracle selection. We then propose PATTS, a novel parallel-aware TTSframework that

searches over intermediate thought-image states using crossmodal verifier signals. PATTS uses a state-dependent reliability weight, derived from model-internal trajectory diagnostics, to modulate the influence ofexternal semantic verification during generation. In thefixed-N = 32 evaluation on MMaDA-Parallel, the complete PATTS configuration reaches a GenEval score of0.844, compared with 0.628 without TTS, 0.787for Best-of-N, and 0.810for FKD. These results show that reliability-aware trajectory verification can improve test-time selectionfor parallel thought-image co-generation under the evaluated search budget. Our code release is at https://github.com/Ringo-Star/PATTS.git

## 1. Introduction

Test-time scaling (TTS) improves generation by spending extra computation at inference time rather than retraining the model. In visual generation, this extra budget is usually allocated along image-centric axes: more denoising steps, more random trajectories, or best-of-N selection with a finalimage VLM verifier model [9, 17, 25]. These strategies are effective when a completed image is the natural unit of search. However, they are less complete for multimodal diffusion language models, where generation can expose a richer trajectory than a sequence of final images.

Conventional CLIP-style prompt-image scores primarily capture global semantic alignment and may fail to distinguish images that satisfy fine-grained compositional constraints from those that do not. On GenEval, CLIP-based search improves the average correctness of a 16-trajectories TTS only from 86.51% to 87.34%, while an oracle selector reaches 96.75%. This substantial gap motivates exploring additional information beyond final-image similarity for guiding test-time search. Parallel thinking-generation models make this limitation explicit. In MMaDA-Parallel [26], textual reasoning tokens and image tokens are decoded jointly inside a single masked diffusion process, as shown in Figure 1. This creates a new inference-time scaling opportunity: instead of verifying the final image, a search method can inspect partially decoded thought-image states and select trajectories using their evolving cross-modal information.

However, exploiting these intermediate states introduces a new challenge: their semantic content may be unreliable before denoising is sufficiently advanced. Early previews can be unstable, while late previews are easier to verify but leave less room to redirect generation. Moreover, the reasoning and image states may become misaligned, producing plausible reasoning traces alongside images that violate spatial relations or fine-grained attributes. The central question is therefore not simply how to sample more candidates, but when an intermediate thought-image state is reliable enough to guide search.

We study this problem in parallel multimodal discrete diffusion LLMs and propose PATTS, a parallel-aware test-time scaling framework for thought-image generation. PATTS maintains multiple denoising trajectories and forms predicted clean thought-image previews at selected cutpoints. Rather than applying external semantic verification uniformly across denoising steps, PATTS estimates the reliability of each intermediate state using lightweight modelinternal diagnostics, include token confidence, uncertainty, preview stability, and cross-modal agreement. A statedependent trust gate then adaptively combines internal diagnostics with external VLM verification, reducing reliance on unstable intermediate previews while emphasizing external verification only in late stage of generation process.

Although intuitive straightforward, our experiments demonstrate the effectiveness of reliability-aware trajectory search. On MMaDA-Parallel, enabling parallel reasoning without verification reduces GenEval score from 0.65 to 0.62, with the largest degradation on position and attribute. Under a fixed N = 32 evaluation, PATTS achieves 0.844, compared with 0.628 without TTS, 0.787 for Best-of-N, and 0.810 for FKD [33]. The fixed-setting component breakdown further shows that trajectory-level search, reward routing, and statedependent trust estimation contribute progressively to the final result, avoiding comparisons across different budgets or configurations.

Our contributions are:

1. We identify empirical limitations of conventional finalimage selection TTS for parallel thought-image generation, including weak alignment between CLIP finalimage scores and compositional correctness, and image semantic degradation under unverified parallel thinking.

2. We propose PATTS, a trajectory-level, reliability-based search framework that uses model-internal reliability diagnostics to weight external VLM verification of partially decoded thought-image states.

3. We demonstrate improved compositional generation quality and inference efficiency over comprehensive text-toimage generation benchmarks and TTS baselines, and ablate the individual contributions of reward routing and state-dependent trust estimation.

## 2. Related Work

Test-time scaling for visual generation. Test-time scaling allocates additional inference compute to improve a frozen generator through sampling, search, or verification. In language models, this idea appears as increased reasoning or search budget [18, 24]; In diffusion image generation, recent methods allocate compute to noise search, evolutionary search, path pruning, or verifier-guided candidate selection [9, 12, 17, 25, 31]. These approaches largely treat the image trajectory as the object of search and score either final samples or visual denoising states. We instead study a different search object: the joint thought-image trajectory exposed by parallel multimodal diffusion LLMs.

Thinking-aware multimodal diffusion models. Thinkingaware image generation asks whether explicit reasoning can improve synthesis and editing. Several recent systems condition generation on a reasoning trace or follow a sequential think-then-generate paradigm [1, 6, 8]. In such pipelines, an error in the thought can propagate into the image, and the image state cannot usually correct the reasoning state once the thought is fixed [30]. Parallel multimodal diffusion models change this interface: MMaDA-Parallel [26] decodes reasoning and image tokens jointly, while unified discrete diffusion generators such as Lumina-DiMOO and Muddit [23, 28] show how masked-token diffusion can support multimodal generation. PATTS uses this parallelism as the testbed for trajectory-level test-time scaling rather than treating the model architecture itself as the contribution.

Verifier-guided multimodal generation. Recent multimodal TTS work extends language-model-style inference scaling to unified multimodal models. However, general vision-language models offer limited verifier reward quality on complex and compositional image generation benchmarks [13]. CLIP-style prompt-image scores [21] are convenient for best-of-N selection, but compositional benchmarks often require object-level counting, attributes, relations, and binding rather than global image–text similarity. Recent multimodal TTS and refinement systems use stronger VLMs or unified models to generate, verify, and revise outputs through multi-round reasoning [4, 14, 29, 32]. These methods mainly verify completed outputs or sequential refinement states. Our focus is on when partially decoded thought-image states in a parallel diffusion trajectory are reliable enough to verify. We find that final prompt-image verifiers can select visually plausible samples while still failing to rank compositional correctness reliably. This motivates treating verification as a trajectory-dependent reliability problem rather than a fixed final-output scoring step.

Intermediate-state verification. Intermediate verification is difficult because intermediate states are not always semantically meaningful. In continuous diffusion models, a verifier often sees a noisy latent or a predicted clean image, so prior work use estimate of final output from current state [9, 17, 25] or train a noise-aware verifier model [31]. Discrete diffusion multimodal LLMs [2] provide a different interface: an intermediate state is partially unmasked, and a clean preview can expose both predicted reasoning tokens and predicted image tokens [19, 23, 28]. This makes prompt-thought, prompt-image, and thought-image consistency measurable before the final sample, but only when the preview is stable enough. Existing intermediate-state verifiers do not exploit this structure and remain insufficient for the partially decoded thought–image states that arise in parallel thinking-aware dMLLMs. We address this reliability problem directly through a state-dependent trust gate rather than applying a fixed verifier schedule.

## 3. Method

## 3.1. Preliminaries

Image-thought parallel multimodal diffusion generation. Following [26], in a multimodal Diffusion LLM, an intermediate state $z _ { t }$ at denoising step t is:

$$
z _ { t } : = ( y _ { t } , v _ { t } \mid h _ { t } ) ,\tag{1}
$$

where $y _ { t }$ are reasoning tokens, $v _ { t }$ are image tokens, $h _ { t }$ refers to model internal information during intermediate timesteps.

The final output is $z _ { 0 } = ( y _ { 0 } , v _ { 0 } )$ . Text and image tokens are denoised in parallel but may follow different schedules, so partially decoded images can be viewed as checkpoints of a finer textual reasoning process. For evaluation of intermediate image tokens $v _ { t } .$ , a clean preview can be formed by filling still-masked positions with the argmax logits. Decoding $\hat { v } _ { 0 } ^ { ( t ) }$ then yields a paired thought-image estimate:

$$
\hat { z } _ { 0 } ^ { ( t ) } = \hat { z } _ { 0 } ( z _ { t } ) = ( \hat { y } _ { 0 } ^ { ( t ) } , \hat { v } _ { 0 } ^ { ( t ) } ) ,\tag{2}
$$

Intermediate reward alignment. Let $p _ { \theta } ( z _ { 0 } )$ be the cleansample distribution of the model θ and $r ( z _ { 0 } )$ a final reward. A KL-regularized reward-aligned target is:

$$
p _ { \mathrm { t a r g e t } } ( z _ { 0 } ) = \frac { 1 } { Z } p _ { \theta } ( z _ { 0 } ) \exp \left( \frac { r ( z _ { 0 } ) } { \alpha } \right) ,\tag{3}
$$

with temperature $\alpha > 0$ . The ideal intermediate potential marginalizes over all clean samples:

$$
V _ { t } ( z _ { t } ) = \alpha \log \mathbb { E } _ { p ( z _ { 0 } | z _ { t } ) } \left[ \exp \left( \frac { r ( z _ { 0 } ) } { \alpha } \right) \right] ,\tag{4}
$$

which is intractable. Practical methods [10, 15] instead use a plug-in score on estimated clean view:

$$
\hat { r } _ { t } ( z _ { t } ) : = r \Big ( \hat { z } _ { 0 } ^ { ( t ) } \Big ) .\tag{5}
$$

However, this surrogate can be biased when previews are unstable, as we have already shown in Figure 1.

## 3.2. PATTS overview

We propose PATTS, a test-time scaling framework for parallel-aware multimodal diffusion language models. Figure 2 illustrates the PATTS pipeline. The key motivation of PATTS is that an intermediate thinking process is usefu only when it can be faithfully estimated and remains aligned with the generated image. Otherwise, early reasoning can drift and hurt compositional accuracy. Therefore, PATTS treats verification as a reliability-weighted search problem: PATTS first converts the partially masked state $z _ { t }$ into a predicted clean thought–image preview $\hat { z } _ { 0 } ^ { ( t ) } = ( \hat { y } _ { 0 } ^ { ( t ) } , \hat { v } _ { 0 } ^ { ( t ) } )$ ). It then computes lightweight internal reliability diagnostics from the current trajectory and uses them to estimate a trust gate $c _ { t } ( z _ { t } )$ . This gate determines how much selection reward should come from internal reliability versus an external VLM verifier applied to the clean preview.

$$
R _ { t } ^ { \mathrm { P A T T S } } ( z _ { t } ) = \left( 1 - c _ { t } ( z _ { t } ) \right) \tilde { r } _ { t } ^ { \mathrm { i n t } } ( z _ { t } ) + c _ { t } ( z _ { t } ) \tilde { r } _ { t } ^ { \mathrm { e x t } } ( z _ { t } ) ,\tag{6}
$$

where $c _ { t } ( z _ { t } ) ~ \in ~ [ 0 , 1 ]$ , and $\tilde { r } _ { t } ^ { \mathrm { i n t } } , \tilde { r } _ { t } ^ { \mathrm { e x t } }$ are ‘standardized rewards. Trust gate values indicate whether the trajectory is confident, stable, and cross-modally coherent enough to be verified. High trust gate values shift verification reward weight to VLM semantic thought-image verification.

![](images/9653fd334e1005a6b77fd38a1e14a9c8fdce679ce01eb3dbf687bbb2530a6f19.jpg)  
Figure 2. Method Overview of PATTS. Left: at each denoising cutpoint, the parallel MdLLM provides token logits, entropy statistics, attention patterns, and an estimated clean preview $\hat { z } _ { 0 } ^ { ( t ) } = ( \hat { y } _ { 0 } ^ { ( t ) } , \hat { v } _ { 0 } ^ { ( t ) } )$ , which are converted into internal reliability features $h _ { t }$ . Then the reliability gate $c _ { t }$ as well as $R _ { t }$ will be calculated. Right: $c _ { t }$ routes reward weight between internal trajectory reliability and an external VLM verifier. Based on reward ranking, trajectories are scored, pruned, and propagated. The final output is selected from the surviving trajectories.

## 3.3. Trust estimation from trajectory reliability

A standard intermediate reward sampler applies a predesigned timestep schedule, which globally suppresses early rewards, but reliability in parallel multimodal diffusion also depends on the prompt, token uncertainty, preview stability, and text–image agreement. In addition, these methods do not give effective verification signals in early generation steps.

PATTS therefore estimates reliability from the current state: In test-time scaling setting, for N trajectories $\{ z _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ , trust is the probability that the intermediate ordering agrees with the final reward ordering:

$$
c _ { t } = \mathrm { P r } \Big [ \mathrm { s i g n } ( \hat { r } _ { t } ^ { ( i ) } - \hat { r } _ { t } ^ { ( j ) } ) = \mathrm { s i g n } ( r _ { 0 } ^ { ( i ) } - r _ { 0 } ^ { ( j ) } ) \Big ] .\tag{7}
$$

This quantity cannot be computed online without completing many trajectories. PATTS therefore uses a state-dependent proxy $( c _ { t } ( z _ { t } ^ { ( i ) } ) )$ , estimated from cheap trajectory features:

$$
\pmb { h } _ { t } ^ { ( i ) } = [ H _ { t } ^ { ( i ) } , M _ { t } ^ { ( i ) } , \rho _ { t } ^ { ( i ) } , S _ { t } ^ { ( i ) } , A _ { t } ^ { ( i ) } , \mathrm { R a n k S t a b } _ { t } ^ { ( i ) } ] .\tag{8}
$$

Here $H _ { t }$ is normalized mean entropy over masked reasoning and image positions, $M _ { t }$ is the mean top-two logit margin, and $\rho _ { t } = | \mathcal { M } _ { t } | / K$ is the remaining mask ratio. The clean preview defines the remaining features: Denoising stability is token overlap between adjacent previews, while cross-modal agreement is cosine similarity in the frozen shared embedding space, $A _ { t } ^ { ( i ) } = \cos ( \bar { E } ( \hat { y } _ { 0 } ^ { ( t ) } ) , \bar { E } ( \hat { v } _ { 0 } ^ { ( t ) } ) )$ . Rank stability measures how much a particle’s preliminary rank changes between adjacent cutpoints, where ranks are computed by $M _ { t } ^ { ( i ) } - H _ { t } ^ { ( i ) } + S _ { t } ^ { ( i ) }$ . The trust gate is

$$
\begin{array} { r } { c _ { t } ( \boldsymbol { z } _ { t } ^ { ( i ) } ) = \sigma \Big ( \omega _ { 0 } + \omega ^ { \top } h _ { t } ^ { ( i ) } \Big ) , } \end{array}\tag{9}
$$

Here, $c _ { t } ^ { ( i ) }$ is used as a state-dependent reliability weight for mixing internal and external scores, rather than as a calibrated probability of final ranking agreement. The feature normalisation and the hyperparameters $( \omega _ { 0 } , \omega , \eta , \tau _ { \mathrm { e x t } } , \beta )$ are fixed on a separate development set and kept fixed throughout evaluation, without prompt-specific tuning.

Algorithm 1 Trust-calibrated intermediate verification   
Require: prompt p; generator p ; verifier Φ; particles N;   
cutpoints $T > t _ { 1 } > \cdot \cdot \cdot > 0$   
1: initialize N particles $\{ z _ { T } ^ { ( i ) } \} _ { i = 1 } ^ { N }$   
2: for each cutpoint t do   
3: for each particle i do   
4: predict clean preview $\hat { z } _ { 0 } ^ { ( i , t ) } = ( \hat { y } _ { 0 } ^ { ( i , t ) } , \hat { v } _ { 0 } ^ { ( i , t ) } )$   
5: compute diagnostics $h _ { t } ^ { ( i ) }$ : entropy, margin, mask   
ratio, stability, agreement, rank stability   
6: estimate trust $\begin{array} { r } { c _ { t } ^ { ( i ) } = \sigma ( \omega ^ { \top } h _ { t } ^ { ( i ) } ) } \end{array}$   
7: compute reward $r _ { t } ^ { \mathrm { i n t } } ( z _ { t } ^ { ( i ) } )$ and uncertainty $u _ { t } ^ { ( i ) }$   
8: if $c _ { t } ^ { ( i ) } \geq \tau _ { \mathrm { e x t } } ;$ external reward $r _ { t } ^ { \mathrm { e x t } } ( z _ { t } ^ { ( i ) } )$ from $\hat { z } _ { 0 } ^ { ( i , t ) }$   
via verifier logits (else reuse cache)   
9: end for   
10: standardize $\boldsymbol { r } _ { t } ^ { \mathrm { i n t } } , \boldsymbol { r } _ { t } ^ { \mathrm { e x t } }$ across particles $\to \tilde { r } _ { t } ^ { \mathrm { i n t } } , \tilde { r } _ { t } ^ { \mathrm { e x t } }$   
11: score $s _ { t } ^ { ( i ) } = \stackrel { \smile } { ( 1 - c _ { t } ^ { ( i ) } ) } \tilde { r } _ { t } ^ { \mathrm { i n t } , ( i \hat { ) } } + c _ { t } ^ { ( i ) } \tilde { r } _ { t } ^ { \mathrm { e x t } , ( i \stackrel {  } { i } ) } - \acute { \beta } u _ { t } ^ { ( i ) }$   
12: keep top- $. K$ by $s _ { t }$   
13: update controller using rank stability and ESS   
14: propagate surviving particles with frozen $p _ { \theta }$   
15: end for   
16: return final candidates $z _ { 0 } .$ , optionally reranked by Φ

## 3.4. Internal and external verification rewards

Internal trajectory reward. When reliability is low, PATTS avoids applying semantic verification to unstable previews

and instead scores trajectory reliability:

$$
r _ { t } ^ { \mathrm { i n t } } ( z _ { t } ) = \eta ^ { \intercal } h _ { t }\tag{10}
$$

with coefficients $\eta .$ This reward does not assert semantic correctness and favors particles likely to become reliable enough for later semantic verification.

External semantic reward. When reliability is high, PATTS evaluates the clean preview with a thinking-aware verifier Φ: The score is read as an affirmative probability from yes/no verifier logits,

$$
r _ { t } ^ { \mathrm { e x t } } ( z _ { t } ) = \frac { e ^ { \ell _ { \mathrm { Y e s } } } } { e ^ { \ell _ { \mathrm { Y e s } } } + e ^ { \ell _ { \mathrm { N o } } } } = \sigma ( \ell _ { \mathrm { Y e s } } - \ell _ { \mathrm { N o } } ) ,\tag{11}
$$

which preserves cross-particle resolution better than decoded integer ratings. Because the verifier is expensive, it is evaluated only when $c _ { t } \geq \tau _ { \mathrm { e x t } } ;$ otherwise the cached score is reused and its contribution remains small through Eq. (6).

Standardized scoring. Internal and external rewards have different scales and variances. PATTS standardizes each reward across the current population before mixing: Thus $c _ { t }$ controls relative influence rather than raw scale. If the external verifier returns nearly identical scores, $\sigma _ { t } ^ { \mathrm { e x t } } \to 0$ and the standardized external term contributes little. Our default implementation keeps the top-K particles by $R _ { t }$ under a linearly decaying budget.

## 3.5. Effective model-internal reliability signals

The internal reward in Eq. (10) is a reliability estimator, not a semantic verifier. Its signs follow from the maskedtoken posterior that produces the clean preview [16]. For each masked position i, let $p ^ { i } = p _ { \theta } ( { \cdot } \mid z _ { t } ) , p _ { \mathrm { m a x } } ^ { i }$ be the preview-token probability, $H ^ { i }$ the predictive entropy, and $M ^ { i } = p _ { ( 1 ) } ^ { i } - p _ { ( 2 ) } ^ { i }$ the top-two margin. Since the maskeddiffusion objective is cross-entropy, preview likelihood is controlled by $\begin{array} { r } { \sum _ { i } \log p _ { \operatorname* { m a x } } ^ { i } } \end{array}$ , and

$$
- \log p _ { \operatorname* { m a x } } ^ { i } \le H ^ { i } , \qquad - \log p _ { \operatorname* { m a x } } ^ { i } \le - \log M ^ { i } .\tag{12}
$$

Low entropy and high margin therefore select previews that are high-likelihood under the frozen denoiser. The margin also has a robustness interpretation: the argmax token is unchanged under any $\ell _ { \infty }$ probability perturbation smaller than $M ^ { \bar { i } } / 2$

Mask ratio and denoising stability control plug-in reward bias. A high $\rho _ { t }$ indicates many unresolved positions; high $S _ { t }$ indicates that adjacent clean previews have converged. If the final verifier is locally smooth in token-space distance, stable previews are more likely to preserve final rankings, matching the trust criterion in Eq. (7). Cross-modal agreement $A _ { t }$ is specific to the parallel multimodal setting: bidirectional attention lets partial reasoning influence image tokens and partial image tokens influence reasoning tokens. Cosine agreement in the shared embedding space is a lightweight proxy for text–image dependence, in the same general form as contrastive mutual-information and CLIP-style alignment objectives [20, 21, 27]. Finally, RankStab directly measures whether particle orderings persist across cutpoints. Together, these signals route selection toward particles that are confident, stable, cross-modally aligned, and likely to become externally verifiable.

## 4. Experiments

## 4.1. Setup

Benchmark and metric. We use the GenEval compositional generation benchmark [7] with 553 prompts spanning six categories: single object, two object, counting, colors, position, and color attributes. Generated images are scored with the standard Mask2Former object detector [5], and we report the GenEval overall score as the unweighted mean of the six category accuracies. We additionally evaluate cross-benchmark robustness on T2I-CompBench [13] and DPG-Bench [11] using each benchmark’s official evaluator and overall score. Unless otherwise annotated, evaluations use 4 repeated runs to decrease evaluation instability.

Generators and configurations. We use the parallel thinking-generation model MMaDA-Parallel [26] as the primary generator. GenEval scores for more diffusion generators [22, 23, 28, 30] are summarized in the supplementary material. Unless stated otherwise, the reference configuration is thinking-aware decoding (32 image denoising steps, 128 text denoising steps, CFG 4.0, at resolution 512×512).

Verifiers. We choose three verifiers in the analysis. For the thinking-aware generator we use a popular VLM model, Qwen2.5-VL-7B-Instruct [3], which scores the prompt against the reasoning text, the final image, or both jointly. A CLIP-style image-prompt score is the conventional choice for best-of-N selection over completed images. A structured matching score (Oracle) that mirrors GenEval’s object-level checks serves as a benchmark-aligned reference.

TTS search methods. We compare final-image selection (best-of-N over completed thought-image pairs) with trajectory-level search that scores the predicted clean state $\hat { z } _ { 0 } ^ { ( t ) }$ at intermediate cut points. PATTS is the trajectory-level variant whose reliability-calibrated trust gate routes verification weight between internal trajectory signals and the external VLM verifier. Unless noted, trajectory-level runs use $N = 3 2$ particles.

## 4.2. Trajectory-gated TTS on MMaDA-Parallel

Overall performance and efficiency. Table 1 compares PATTS with representative test-time scaling baselines under MMaDA-Parallel. PATTS achieves the best overall

![](images/bb37123b49cf299332a29894c8719ba99fa2d6d16f6d6b3d8f5dfdb489631579.jpg)

Table 1. Quantitative performance comparison on GenEval with MMaDA-Parallel. NFE counts forward passes through the image generator; Ver. counts verified candidate previews per prompt. Latency is end-to-end wall-clock latency per generated image. We report error bars only for the overall score. Best results and second-best results are annotated for each column. † reports re-implementation from original baseline on MMaDA-Parallel.
<table><tr><td>Method</td><td>NFE</td><td>Ver.</td><td>Latency</td><td>GenEval ↑</td><td>Single</td><td>Two</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attributes</td></tr><tr><td>No TTS</td><td>161</td><td>0</td><td>9.9</td><td> $0 . 6 2 8 { \scriptstyle \pm . 0 3 7 }$ </td><td>0.975</td><td>0.657</td><td>0.506</td><td>0.777</td><td>0.425</td><td>0.440</td></tr><tr><td>HTS</td><td>1796</td><td>62</td><td>58.3</td><td> $0 . 7 1 1 { \scriptstyle \pm . 0 2 3 }$ </td><td>0.988</td><td>0.798</td><td>0.594</td><td>0.792</td><td>0.430</td><td>0.661</td></tr><tr><td>Best-of-N / LTS</td><td>5184</td><td>32</td><td>145.7</td><td> $0 . 7 8 7 { \scriptstyle \pm . 0 1 7 }$ </td><td>0.988</td><td>0.806</td><td>0.713</td><td>0.906</td><td>0.630</td><td>0.680</td></tr><tr><td>FKD [33]†</td><td>5280</td><td>160</td><td>154.9</td><td> $\underline { { 0 . 8 1 0 } } \pm . 0 1 8$ </td><td>1.000</td><td>0.859</td><td>0.707</td><td>0.894</td><td>0.690</td><td>0.710</td></tr><tr><td>PATTS (ours)</td><td>1796</td><td>38.2</td><td>55.1</td><td> $\mathbf { 0 . 8 4 4 _ { \pm . 0 1 4 } }$ </td><td>1.000</td><td>0.899</td><td>0.753</td><td>0.915</td><td>0.718</td><td>0.776</td></tr></table>

Table 2. Component breakdown of PATTS on GenEval with MMaDA-Parallel. The rows progressively add joint thoughtimage verification, reward routing, and trust estimation. Bold marks the complete method. $N = 3 2$ on full GenEval benchmark.
<table><tr><td>Configuration</td><td>TTS</td><td>Trust Gate Verifier</td><td></td><td>Reward</td><td>GenEval↑</td></tr><tr><td>No TTS</td><td>x</td><td>一</td><td></td><td></td><td>0.628</td></tr><tr><td>+ HTS</td><td>√</td><td>1</td><td>txt + img external</td><td></td><td>0.711</td></tr><tr><td>→ Trust-Gated Beam</td><td>√</td><td>fixed phase</td><td> $\mathrm { \ t x t { + i m g } }$ </td><td>external</td><td>0.773</td></tr><tr><td>+ hard two-phase reward</td><td>√</td><td> $\operatorname { f i x e d p h a s e }$ </td><td> $\mathrm { \ t x t { + i m g } }$ </td><td>hard switch</td><td>0.795</td></tr><tr><td>+ timestep trust gate  $c _ { t } : = t ~ \checkmark$ </td><td></td><td> $\mathrm { \ t i m e s t e p }$ </td><td> $\operatorname { t x t } + \operatorname { i m g }$ </td><td>mix</td><td>0.812</td></tr><tr><td>→ full trust gate (PATTS) √</td><td></td><td> $c _ { t } ( h _ { t } )$ </td><td> $\mathrm { \ t x t { + i m g } }$ </td><td>mix</td><td>0.844</td></tr></table>

GenEval score of 0.844, improving over No TTS (0.628), HTS (0.711), Best-of-N/LTS (0.787), and FKD (0.810). Relative to the strongest baseline, FKD, PATTS improves or keeps accuracy across all subtasks. PATTS is also substantially more compute-efficient, with 1796 generator NFEs, compared with 5184 for Best-of-N and 5280 for FKD, while reducing end-to-end latency from 154.9s for FKD to 55.1s. These results indicate that the gain does not arise simply from generating or verifying more candidates, but from allocating test-time computation more selectively along the joint thought-image trajectory.

Performance gain from PATTS components. Table 2 provides a controlled ablation of the complete PATTS result on the full GenEval benchmark with $N = 3 2$ . Starting from the no-TTS score of 0.628, applying trajectory-level HTS with joint text-image verification increases performance to 0.711. Introducing trust-gated beam further raises the score to 0.773. Replacing direct external verification with a hard two-phase internal/external reward switch gives 0.795. A continuous reward mixture controlled only by the denoising timestep, $c _ { t } : = t ,$ , reaches 0.812. Finally, replacing this fixed timestep schedule with the proposed state-dependent trust $c _ { t } ( h _ { t } )$ increases performance to 0.844. Trajectory-level search, reward routing, and state-dependent trust estimation contribute successively to the final result of overall $0 . 6 2 8  0 . 8 4 4$ improvement. In particular, holding the verifier input and continuous reward mixture fixed, replacing timestep-only trust with $c _ { t } ( h _ { t } )$ gives a 3.2 percentage-point improvement,

directly isolating the contribution of the proposed trust gate.  
![](images/a54bfbe2ec98bf0a59b5f2c171ce2db52325e7f7b23fd6e116d87e3a3a93c7e9.jpg)  
Figure 3. Cross-benchmark comparison of test-time scaling methods. Overall scores on GenEval, T2I-CompBench, and DPG-Bench under the same MMaDA-Parallel backbone; error bars denote standard deviation from four runs.

Cross-benchmark robustness. Figure 3 evaluates whether the advantage persists beyond GenEval on two additional compositional-generation benchmarks. PATTS obtains the highest mean score on GenEval, T2I-CompBench, and DPG-Bench. Together with the error bars, these results provide evidence that reliability-weighted trajectory allocation is not specific to the GenEval evaluator or to a single compositional category. Because all full PATTS experiments use the MMaDA-Parallel backbone, we interpret this as benchmarklevel robustness rather than model-level generalization.

Scaling with test-time compute. Figure 4 further investigates PATTS trajectory-search scaling versus baselines on the GenEval counting subset (we provide further details in the supplementary material). Across the evaluated budgets, PATTS improves as the particle budget increases and remains competitive with or above external-only trajectory selection at matched generation cost. The curves flatten at larger budgets, indicating diminishing returns from additional candidates. This behavior is consistent with the verifier-alignment analysis in Sec. 4.3: once the candidate pool contains sufficiently strong samples, further scaling becomes increasingly limited by the ability of the verifier to rank them rather than by candidate generation alone.

![](images/b58c8a87deb1bb7b6cbe4d453392ea0514a59963f91952331d1f800f145d3825.jpg)  
Figure 4. Test-time compute scaling on the GenEval counting subset. GenEval score versus model-generation NFEs per prompt for MMaDA-Parallel. PATTS remains competitive with externalonly selection across the evaluated budgets, while both curves flatten as the budget grows.

Table 3. MMaDA-Parallel T2I vs. thinking-aware generation.
<table><tr><td>Setting</td><td>Single</td><td>Two</td><td>Count</td><td>Color</td><td>Pos.</td><td>Attr.</td><td>Overall</td></tr><tr><td>Thinking-Aware</td><td>0.97</td><td>0.72</td><td>0.48</td><td>0.70</td><td>0.31</td><td>0.54</td><td>0.62</td></tr><tr><td>Standard T2I (w/o thinking)</td><td>0.95</td><td>0.71</td><td>0.43</td><td>0.70</td><td>0.57</td><td>0.56</td><td>0.65</td></tr><tr><td>∆ (think − std)</td><td>0.02</td><td>0.01</td><td>0.05</td><td>0.00</td><td>-0.26</td><td>-0.02</td><td>-0.03</td></tr></table>

## 4.3. Two failure modes behind the comparison

The performance pattern in Sec. 4.2 is explained by two properties of final-image test-time scaling that trajectory gating is designed to address.

Failure mode 1: final-image verifiers are weakly aligned with compositional correctness. A final-image verifier can raise selected-image accuracy by preferring visually plausible samples while remaining poorly calibrated to compositional correctness. We test this on Lumina-DiMOO beamsearch outputs (553 prompts, $N = 1 6 )$ : for each prompt we compare the candidate-pool average, the CLIP-selected image, and an oracle that succeeds if any candidate for the prompt is correct (Fig. 5). The candidate pool is already strong (86.51% correct on average); CLIP selection reaches 87.34%, only 0.83 points above the candidate average, while the oracle reaches 96.75%. Most of the selected accuracy therefore originates in the candidate pool rather than in verifier ranking, and the 9.41-point gap to the oracle bounds the headroom available to a better-aligned verifier. This gap is largest for attribute prompts (16.0 points).

The score-level statistics are consistent with this reading. For ranking candidates within a prompt, the CLIP score has ROC-AUC 0.464, point-biserial correlation $r { = } - 0 . 0 4 6 .$ , and a mean score gap of −0.005 between correct and incorrect candidates; on two-object prompts it is anti-correlated with correctness (AUC 0.382). The weakness is specific to the final-image CLIP score rather than to verification in general: a structured matching score that mirrors GenEval’s object-level checks separates correct from incorrect images substantially better. The same limitation is visible on SD 1.5, detailed in supplementary material.

![](images/db93c956c8614363bde22e5ac1c246e8dd16903e86826936df9e6f037c33e61d.jpg)  
Figure 5. Per-category verifier headroom on Lumina-DiMOO $N = 1 6$ beam-search outputs. Bars show candidate-pool average correctness, CLIP-selected correctness, and the oracle success rate when any candidate for the prompt is correct.

Failure mode 2: unverified parallel thinking can introduce semantic drift. Table 3 shows that enabling the reasoning trace without verification slightly reduces overall GenEval accuracy, from 0.65 to 0.62. The degradation is not uniform: single-object, two-object, counting, and color prompts are largely preserved or slightly improved, while position and attribute prompts degrade. This suggests that the reasoning trace does not simply reduce generation quality globally. Instead, it can perturb fine-grained semantic binding and spatial grounding in cases where the text and image states are not well aligned.

PATTS reward correlation with GenEval correctness  
![](images/e705912c846e5870dcfb3e661027f7f429d93a50d8c9977d25a8b67007d9e823.jpg)  
image denoising cutpoint (128 text thinking steps)  
Figure 6. Correlation Test between GenEval correctness and reward signals. We Spearman-correlate every selection signal against whether that prompt’s final image is GenEval-correct $( N =$ 16, cutpoints shown at text-thinking step t of 128).

Table 4. Ablation study of PATTS designs. “RankAcc”: intermediate-final ranking agreement, “Regret”: loss from selecting a suboptimal intermediate state, “Score/Call”: verifier-call efficiency. Defaults are shown in the bottom row. Evaluated on GenEval counting subset at $N = 1 6$
<table><tr><td>Guidance Signal</td><td>GenEval↑</td></tr><tr><td>no TTS</td><td>0.50</td></tr><tr><td>external only</td><td>0.62</td></tr><tr><td>internal only</td><td>0.45</td></tr><tr><td>hard two-phase</td><td>0.58</td></tr><tr><td>reliability mixture</td><td>0.66</td></tr></table>

(a) Reward routing. We compare external / internal trajectory verifica tion, hard phase switching, and the proposed mixture.

<table><tr><td>Trust Features</td><td>RankAcc↑</td></tr><tr><td>timestep only</td><td>0.17</td></tr><tr><td> $H _ { t } , M _ { t } , \rho _ { t }$ </td><td>0.02</td></tr><tr><td> $+ S _ { t }$   $+ A _ { t }$ </td><td>0.07 0.13</td></tr><tr><td></td><td></td></tr><tr><td> $+ { \mathrm { R a n k S t a b } } _ { t }$ </td><td>0.19</td></tr></table>

(b) Trust estimation. We ablate the state-dependent trust gate c $\left( z _ { t } \right)$ from timestep-only tempering to full multitrajectory reliability estimation.

<table><tr><td>Selection Score</td><td>Regret↓</td></tr><tr><td> $\boldsymbol { r } _ { t } ^ { \mathrm { e x t } }$ </td><td>0.21</td></tr><tr><td> $c _ { t } r _ { t } ^ { \mathrm { e x t } }$ </td><td>0.20</td></tr><tr><td> $r _ { t } ^ { \mathrm { i n t } }$ </td><td>0.34</td></tr><tr><td> $R _ { t } ^ { \mathrm { { ^ { c } A T T S } } }$ </td><td>0.22</td></tr><tr><td> $R _ { t } ^ { \mathrm { P A T T S } }$   $\beta u _ { t }$ </td><td>0.19</td></tr></table>

(c) Particle scoring. We test whether the internal fallback and penalty improve selection decisions. Regret is vs. a held-out terminal-verifier oracle.

## 4.5. Ablations

## 4.4. When is a state reliable enough to verify?

Table 4 isolates the PATTS components on the GenEval at N = 16 on counting subset, where the internal trajectory reward is least predictive of final correctness and therefore provides a challenging test for the method. We report mixed outcomes: For reward routing, the continuous reliability mixture improves over external-only verification from 0.62 to 0.66, while internal-only routing falls below no-search; on this subset the internal reward is thus useful as a gate input but not as a standalone selector. For trust estimation, the full state-dependent feature set slightly outperforms a timestep-only schedule (rank accuracy 0.19 vs. 0.17), while partial feature subsets remain below the timestep baseline. The clearest gain appears in verifier efficiency: the adaptive controller raises score per verifier call from 0.021 to 0.036, substantially reducing verifier cost relative to a fixed beam. Therefore, on this subset, the main measurable benefit of the reliability-calibrated design is improved verifier-call

<table><tr><td>Controller</td><td>Score/Call↑</td></tr><tr><td>fixed top-K</td><td>0.021</td></tr><tr><td>fixed λt</td><td>0.034</td></tr><tr><td>ESS control</td><td>0.027</td></tr><tr><td>rank-stab.</td><td>0.031</td></tr><tr><td>adaptive</td><td>0.036</td></tr></table>

(d) Search controller. We isolate each online update $( \lambda _ { t } ,$ uncertainty β, ESS resampling) over a fixed top-K baseline with identical reward routing.

PATTS applies the external verifier to the predicted clean state $\hat { z } _ { 0 } ^ { ( t ) }$ only when the trajectory is estimated to be reliable, so whether such a moment is identifiable from cheap online features is an empirical question. In Figure 6, under hierarchical beam search, at each cut point we take the top-scoring particle per prompt and Spearman-correlate each selection signal against whether that prompt’s final image is GenEval-correct. For external VLM reward, the correlations are relatively weak and unstable at early cut points and become stronger and more consistent as denoising proceeds. This is the regime the trust gate is designed to exploit: it down-weights external verification while the predicted clean state is uninformative and increases its weight once the state stabilizes. The same observation bounds what the gate can achieve on attribute prompts, the category where the late-denoising signal remains weakest (failure mode 1) and where, correspondingly, trust-gated verification yields the smallest gain in Sec. 4.2.

efficiency rather than a further gain in absolute accuracy, consistent with the weak intermediate-reward correlation reported in Sec. 4.4.

Relation between trust estimation and ranking accuracy. The modest ranking advantage of the trust features is consistent with the end-to-end results. In Table 4b, the full features improve intermediate-final rank agreement from 0.17 to 0.19 over timestep-only trust. This measures a local ranking property on the N = 16 counting subset, whereas Table 2 reports terminal accuracy on the full benchmark after repeated trajectory selection. The result therefore suggests that trajectory diagnostics provide a modest improvement over timesteponly reliability estimation, while the larger end-to-end gain accumulates across trajectory search, reward routing, and trust-aware selection.

## 5. Conclusion

We studied test-time scaling for parallel multimodal diffusion LLMs, where reasoning tokens and image tokens co-evolve within a single generation trajectory. Our analysis shows that final-image selection alone can be limited in this setting: CLIP-style prompt-image scores may favour visually plausible samples without reliably ranking compositional correctness. We introduce PATTS, a statedependent reliability-weighted trajectory search method that uses model-internal diagnostics to determine when predicted clean thought-image states should receive stronger external VLM verification. On GenEval, the complete PATTS configuration improves over final-output reranking in the fixed-N = 32 design sweep. The ablations further show that reliable intermediate verification remains the central limitation: internal trajectory signals are not semantic verifiers by themselves, and future progress depends on stronger verifier alignment and more reliable intermediate-state scoring.

## Acknowledgments

This research is funded in part by an ARC (Australian Research Council) Discovery Grant of DP220100800. Prof. Hongdong Li holds concurrent appointments as a Full Professor with the ANU and as an Amazon Scholar with Amazon (part time). This paper describes work performed at ANU and is not associated with Amazon.

## References

[1] Inclusion AI, Tiwei Bie, Haoxing Chen, Tieyuan Chen, Zhenglin Cheng, Long Cui, Kai Gan, Zhicheng Huang, Zhenzhong Lan, Haoquan Li, et al. Llada2. 0-uni: Unifying multimodal understanding and generation with diffusion large language model. arXiv preprint arXiv:2604.20796, 2026. 2

[2] Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces, 2021. 3

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5- vl technical report, 2025. 5

[4] Leon Liangyu Chen, Haoyu Ma, Zhipeng Fan, Ziqi Huang, Animesh Sinha, Xiaoliang Dai, Jialiang Wang, Zecheng He, Jianwei Yang, Chunyuan Li, et al. Unit: Unified multimodal chain-of-thought test-time scaling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 30257–30267, 2026. 3

[5] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022. 5

[6] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025. 2

[7] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-toimage alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 5

[8] Ziyu Guo, Renrui Zhang, Chengzhuo Tong, Zhizheng Zhao, Peng Gao, Hongsheng Li, and Pheng-Ann Heng. Can we generate images with cot? let’s verify and reinforce image generation step by step. arXiv preprint arXiv:2501.13926, 2025. 2

[9] Haoran He, Jiajun Liang, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, and Ling Pan. Scaling image and video generation via test-time evolutionary search, 2025. 2, 3

[10] Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Dongjun Kim, Wei-Hsiang Liao, Yuki Mitsufuji, J Zico Kolter, Ruslan Salakhutdinov, and Stefano Ermon. Manifold preserving guided diffusion. In The Twelfth

International Conference on Learning Representations, 2024. 3

[11] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment, 2024. 5

[12] Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. Efficient test-time scaling via selfcalibration. In NeurIPS 2025 Workshop on Efficient Reasoning, 2025. 2

[13] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for openworld compositional text-to-image generation, 2023. 3, 5

[14] Zheng Jiang, Yiming Chen, Nan He, Jiahui Chen, Chaoyang Li, Houde Qian, and Lifeng Sun. Test-time scaling over perception: Resolving the grounding paradox in thinking with images, 2026. 3

[15] Sunwoo Kim, Minkyu Kim, and Dongmin Park. Testtime alignment of diffusion models without reward overoptimization. arXiv preprint arXiv:2501.05803, 2025. 3

[16] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution, 2024. 5

[17] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, et al. Inference-time scaling for diffusion models beyond scaling denoising steps. arXiv preprint arXiv:2501.09732, 2025. 2, 3

[18] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori B Hashimoto. s1:\` Simple test-time scaling. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20275–20321, 2025. 2

[19] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025. 3

[20] Ben Poole, Sherjil Ozair, Aaron van den Oord, Alexander A. Alemi, and George Tucker. On variational bounds of mutual information, 2019. 5

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 3, 5

[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 5, 2

[23] Qingyu Shi, Jinbin Bai, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Shuangyong Song, Yunhai Tong, Xiangtai Li, Xuelong Li, et al. Muddit: Liberating generation beyond text-to-image with a unified discrete diffusion model. arXiv preprint arXiv:2505.23606, 2025. 2, 3, 5

[24] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more

effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024. 2

[25] Vignesh Sundaresha, Akash Haridas, Vikram Appia, and Lav R. Varshney. An efficient test-time scaling approach for image generation. In ICLR 2026 2nd Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, 2026. 2, 3

[26] Ye Tian, Ling Yang, JiongFan Yang, Anran Wang, Yu Tian, Jiani zheng, Haochen Wang, Zhiyang Teng, Zhuochen Wang, Yinjie Wang, Yunhai Tong, Mengdi Wang, and Xiangtai Li. MMaDA-parallel: Multimodal large diffusion language models for thinking-aware editing and generation. In The Fourteenth International Conference on Learning Representations, 2026. 2, 3, 5

[27] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019. 5

[28] Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025. 2, 3, 5

[29] Yi Xin, Siqi Luo, Tianxiang Xu, Qi Qin, Haoxing Chen, Kaiwen Zhu, Zhiwei Zhang, Yangfan He, Rongchao Zhang, Jinbin Bai, et al. dmllm-tts: Self-verified and efficient testtime scaling for diffusion multi-modal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 35726–35735, 2026. 3

[30] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. Advances in Neural Information Processing Systems, 38:138867–138907, 2026. 2, 5

[31] Qingtao Yu, Changlin Song, Minghao Sun, Zhengyang Yu, Vinay Kumar Verma, Soumya Roy, Sumit Negi, Hongdong Li, and Dylan Campbell. Ttsnap: Test-time scaling of diffusion models via noise-aware pruning. arXiv preprint arXiv:2511.22242, 2025. 2, 3

[32] Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, and Yujiu Yang. Generative universal verifier as multimodal meta-reasoner. arXiv preprint arXiv:2510.13804, 2025. 3

[33] XiangCheng Zhang, Haowei Lin, Haotian Ye, James Zou, Jianzhu Ma, Yitao Liang, and Yilun Du. Inference-time scaling of diffusion models through classical search. In The Fourteenth International Conference on Learning Representations, 2026. 2, 6

# PATTS: Parallel-Aware Test-Time Scaling for Thought-Image Generation in Multimodal Discrete Diffusion LLMs

Supplementary Material

## A. PATTS Reliability Computation

Cutpoints and clean previews. MMaDA-Parallel decodes reasoning tokens over $T { = } 1 2 8$ steps and updates image tokens on a sparser grid of 32 steps, evenly spaced from text step 32 to 127. A cutpoint is right after the m-th image update. We use m ∈ {2, 4, 8, 16} and obtain the final selection at m=32. At each cutpoint, one forward pass of the frozen generator gives logits for every position. Each still-masked position is filled with its argmax token, which gives the clean preview $\hat { z } _ { 0 } ^ { ( t ) } = ( \hat { y } _ { 0 } ^ { ( t ) } , \hat { v } _ { 0 } ^ { ( t ) } )$ . Positions that are already unmasked keep their tokens.

Token statistics $( H _ { t } , M _ { t } , \rho _ { t } )$ . Let $\mathcal { M } _ { t }$ be the set of positions that are still masked, pooled over the reasoning span and the image span. Let L be the total number of reasoning and image positions. For $j \in \mathcal { M } _ { t }$ , let $p ^ { j }$ be the softmax over that position’s vocabulary $\gamma _ { j } { \mathrm { : } }$ : the text vocabulary for reasoning positions and the VQ codebook for image positions. Then

$$
\begin{array} { r l } & { H _ { t } = \displaystyle \frac { 1 } { | \mathcal { M } _ { t } | } \sum _ { j \in \mathcal { M } _ { t } } \frac { - \sum _ { v } p _ { v } ^ { j } \log p _ { v } ^ { j } } { \log | \mathcal { V } _ { j } | } , } \\ & { M _ { t } = \displaystyle \frac { 1 } { | \mathcal { M } _ { t } | } \sum _ { j \in \mathcal { M } _ { t } } \big ( p _ { ( 1 ) } ^ { j } - p _ { ( 2 ) } ^ { j } \big ) , } \\ & { \rho _ { t } = \displaystyle \frac { | \mathcal { M } _ { t } | } { L } , } \end{array}
$$

where $p _ { ( 1 ) } , p _ { ( 2 ) }$ are the two largest probabilities. If no position is masked, $H _ { t }$ and $M _ { t }$ become the average of the text and image means over all positions.

Denoising stability $( S _ { t } )$ . Let ov $\begin{array} { r } { ( a , b ) = \frac { 1 } { | a | } \sum _ { j } { \rlap / } { \rlap / } { b } / [ a _ { j } = b _ { j } ] } \end{array}$ be the fraction of positions where two token sequences agree. The previous preview is that of the same particle, which survived the previous cut. Then

$$
\begin{array} { r } { S _ { t } ^ { ( i ) } = \frac { 1 } { 2 } \mathrm { o v } ( \hat { y } _ { 0 } ^ { ( t ) } , \hat { y } _ { 0 } ^ { ( t - \Delta ) } ) + \frac { 1 } { 2 } \mathrm { o v } ( \hat { v } _ { 0 } ^ { ( t ) } , \hat { v } _ { 0 } ^ { ( t - \Delta ) } ) , } \end{array}\tag{13}
$$

Here $t - \Delta$ denotes the previous cutpoint, and $S _ { t } ^ { ( i ) } = 0$ at the first cutpoint.

Rank stability (RankStab<sub>t</sub>). Given each trajectory with a provisional score $q _ { t } ^ { ( i ) } = M _ { t } ^ { ( i ) } - H _ { t } ^ { ( i ) } + S _ { t } ^ { ( \bar { i } ) }$ . Let B be the current population size. rank $\mathbf { \Phi } _ { t } ^ { ( i ) } \in \{ 0 , \dots , B - 1 \}$ is particle i’s rank by $q _ { t }$ among the $B$ particles, and $\mathrm { r a n k } _ { t - \Delta } ^ { ( i ) }$ is its rank by the scores $q _ { t - \Delta }$ that the same B particles had at the previous cutpoint. Because both ranks are taken over the same population, pruning does not count as instability.

Then

$$
\mathrm { R a n k S t a b } _ { t } ^ { ( i ) } = \mathrm { e x p } \left( - \frac { \vert \mathrm { r a n k } _ { t } ^ { ( i ) } - \mathrm { r a n k } _ { t - \Delta } ^ { ( i ) } \vert } { B } \right) ,\tag{14}
$$

and $\mathrm { R a n k S t a b } _ { t } ^ { ( i ) } = 0$ at the first cutpoint.

The external reward combines three Qwen2.5-VL-7B answers on image-thjought questions, each scored as $\sigma ( \ell _ { \mathrm { Y e s } } -$ $\ell _ { \mathrm { N o } } )$ . A particle is verified only when $c _ { t } ^ { ( i ) } \geq \tau _ { \mathrm { e x t } }$ . Otherwise it keeps the score from its last verification. Particles that have never been verified get $\tilde { r } ^ { \mathrm { e x t } } = 0$ , the population mean. Both rewards are standardized over the current population,

$$
\tilde { r } ^ { ( i ) } = \frac { r ^ { ( i ) } - \mu _ { t } } { \sigma _ { t } + \epsilon } , \qquad \epsilon = 1 0 ^ { - 6 } ,\tag{15}
$$

where $\left( \mu _ { t } , \sigma _ { t } \right)$ for $r ^ { \mathrm { e x t } }$ are computed over verified particles only. We keep the top- $K _ { t }$ particles by this score, with $K _ { t } =$ 16, 8, 4, 2 at the four cutpoints.

## B. Additional Experiment Results

Example of parallel thought-image generation. Figure B shows one generated complete reasoning trace together with the image tokens committed at different denoising intervals. The example illustrates the setting studied in the main paper: reasoning and image content are produced jointly, so the intermediate trajectory contains cross-modal information before the final image is decoded.

Baseline Compositional Accuracy. Table A reports the baseline GenEval evaluations used to contextualize the main experiments. Lumina-DiMOO is the strongest generator in this group, reaching an overall score of 0.7824 and improving substantially over SD 1.5 on relation-heavy categories such as position and color attributes. MMaDA is competitive on single-object and color prompts but remains weak on position and attribute binding. Muddit is substantially below the other models in this run; the generation logs indicate that this configuration used too few denoising steps, so we treat it as an under-tuned baseline rather than a definitive estimate of the model family.

Selection-based test-time scaling works on conventional generators, but is verifier-limited. Table B shows that independent sampling followed by verifier selection is a strong baseline for a conventional generator. Best-of-N improves SD 1.5 from 0.4239 to 0.5392 at $N = 1 6$ and

<table><tr><td>Model</td><td>Single</td><td>Two</td><td>Count</td><td>Color</td><td>Pos.</td><td>Attr.</td><td>Overall</td></tr><tr><td>SD 1.5 [22]</td><td>95.94</td><td>39.39</td><td>36.88</td><td>74.20</td><td>3.25</td><td>4.29</td><td>0.4232</td></tr><tr><td>Lumina-DiMOO [28]</td><td>98.75</td><td>82.58</td><td>63.44</td><td>86.44</td><td>74.00</td><td>64.25</td><td>0.7824</td></tr><tr><td>MMaDA [30]</td><td>91.87</td><td>51.01</td><td>29.38</td><td>70.48</td><td>14.25</td><td>11.25</td><td>0.4471</td></tr></table>

Table A. Baseline GenEval accuracy. Category columns report accuracy in percent; overall is the average over categories.

0.5444 at $N \ = \ 3 2$ The gains are concentrated in twoobject, counting, color, position, and color-attribute prompts, while single-object prompts are already near saturation. The hierarchical HTS variant improves less, reaching 0.4566 at N=32, K=8; in this setting, allocating compute to a larger set of independent samples is more effective than the beamstyle allocation. Consistent with failure mode 1 in the main paper, these CLIP-guided runs improve the selected samples but still leave substantial compositional headroom relative to an oracle selector.

<table><tr><td>Method</td><td>N</td><td>K</td><td>Single</td><td>Two</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attribute</td><td>Overall</td></tr><tr><td>Baseline</td><td>1</td><td>一</td><td>97.50</td><td>37.37</td><td>35.00</td><td>74.47</td><td>7.00</td><td>3.00</td><td>0.4239</td></tr><tr><td>LTS</td><td>16</td><td></td><td>98.75</td><td>71.72</td><td>50.00</td><td>84.04</td><td>10.00</td><td>9.00</td><td>0.5392</td></tr><tr><td>LTS</td><td>32</td><td>一</td><td>100.00</td><td>72.73</td><td>50.00</td><td>81.91</td><td>11.00</td><td>11.00</td><td>0.5444</td></tr><tr><td>HTS</td><td>16</td><td>4</td><td>92.50</td><td>57.58</td><td>31.25</td><td>63.83</td><td>9.00</td><td>4.00</td><td>0.4303</td></tr><tr><td>HTS</td><td>32</td><td>8</td><td>91.25</td><td>56.57</td><td>46.25</td><td>64.89</td><td>8.00</td><td>7.00</td><td>0.4566</td></tr></table>

Table B. SD 1.5 test-time scaling on GenEval. All methods return one selected image for each of the 553 prompts. Category columns report accuracy in percent.

Unverified thinking degrades compositional accuracy. Table 3 in the main paper isolates the effect of enabling the reasoning trace under the active MMaDA-Parallel evaluation used in the main paper. Standard T2I decoding without the reasoning trace reaches a GenEval overall score of 0.65, while thinking-aware decoding without verification reaches 0.62 (−0.03). The change is category-dependent: single-object, two-object, and counting accuracy are preserved or slightly improved (0.95 → 0.97, 0.71 → 0.72, and 0.43 → 0.48), color is unchanged, but position accuracy drops sharply (0.57 → 0.31) and attribute accuracy decreases slightly (0.56 → 0.54). This supports the mainpaper interpretation that unverified parallel thinking does not simply reduce image quality globally; rather, it can introduce semantic drift in spatial grounding and fine-grained binding when the text and image states are not kept aligned.

Scaling with candidate budget. For the test-time compute scaling experiment in Figure 4, all methods are evaluated on the GenEval counting subset with MMaDA-Parallel using 128 text-diffusion steps and 32 image diffusion steps. N denotes the candidate or particle budget, and NFE denotes the mean number of image-generator forward passes per prompt. Best-of-N independently generates and verifies N completed candidates. PATTS w/o Trust Gate performs intermediate trajectory selection using external verification only, while PATTS uses the full reliability-gated search. The no-TTS baseline uses 161 NFEs, requiring no verifier calls.

![](images/3e95baf5b941507a1861c69215c483a04cb7f0d7eaf77f7c7e90825fd578664c.jpg)

![](images/fad936a83bd1f54fdf6bf56dbc2eb027c203fb62ed07d1eae862fafe7661adfb.jpg)  
Figure A. Prompt: a photo of a bench Color map: steps 0-3 steps 4-7 steps 8-11 steps 12-15  
To transform the question image into the answer image, I need to focus on the following aspects: 1. Lighting: The current image has shadows and appears somewhat dimmer compared to the answer image. The answer image has a brighter, more even lighting setup, suggesting an overcast or early morning light, which can enhance the mood and create a more serene atmosphere . 2. Color Tone : The colors in the question image are muted and somewhat desaturated . The answer image has a vibrant blue tint, which could be due to the sunrisecast sunrise. This will adds a sense of warmth and and a feeling of tranquility to the scene. 3. Background: The background in the question image is blurred and slightly out of focus . The answer image has a shallow depth of field, focusing on the bench andurring out the field. This can create a sense of depth and draw attention to the contrast between the bench and the background . 4. and Texture : The wood in the answer image looks slightly more worn and aged compared to the clean, new bench in the question image . This can imply a passage over time, enhancing the realism of the scene. By applying these adjustments, the image image will better reflect the transformation.

## C. Additional Visual Examples

Figure A illustrates the full reasoning trace in MMaDA-Parallel. In Figure B, Best-of-N and FKD rank candidates by their final or intermediate verifier scores and can prefer images that look plausible but miss the compositional constraint: a black object without the bun, a white cap fused into the hydrant instead of a separate vase, or an extra white sheep. PATTS weights the verifier by the trust gate and selects trajectories whose previews stay stable and cross-modally consistent, which here preserves the counted, placed, and attribute-bound objects.

a photo of a black hot dog (colors)

a photo of four clocks (counting)

a photo of a vase above a fire

hydrant (position)

a photo of a cat below a backpack (position)

a photo of a white bottle and a blue sheep (attribute)

![](images/011cf030b025e666fd8576dbb8eb892323d88cb5ad0625e2be7f85a489b71513.jpg)  
Figure B. Qualitative comparison with TTS baselines. Final images selected by each method for GenEval prompts with MMaDA-Parallel (N=32, 128 text / 32 image denoising steps, CFG 4.0). All methods select from the same 32 recorded thought-image trajectories per prompt (FKD’s resampling is replayed on this pool); No TTS shows a random trajectory. ✓/✗ is the GenEval detector’s prediction.