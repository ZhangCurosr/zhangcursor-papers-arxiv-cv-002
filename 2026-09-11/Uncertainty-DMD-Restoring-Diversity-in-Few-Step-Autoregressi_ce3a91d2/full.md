0s

# Uncertainty DMD: Restoring Diversity in Few-Step Autoregressive Video Distillation

ZIXUAN DUAN<sup>∗</sup>, Nanjing University, Institute of Artificial Intelligence, China Telecom (TeleAI), China

XUNZHI XIANG<sup>∗</sup>, Nanjing University, Institute of Artificial Intelligence, China Telecom (TeleAI), China

YABO CHEN<sup>∗</sup>, Institute of Artificial Intelligence, China Telecom (TeleAI), China

XIN ZHANG, Fudan University, Institute of Artificial Intelligence, China Telecom (TeleAI), China

CHANGHAN LIU, Nanjing University, China

HAIBIN HUANG, Institute of Artificial Intelligence, China Telecom (TeleAI), China

CHI ZHANG, Institute of Artificial Intelligence, China Telecom (TeleAI), China

QI FAN<sup>†</sup>, Nanjing University, China

XUELONG LI<sup>†</sup>, Institute of Artificial Intelligence, China Telecom (TeleAI), China

Self forcing  
![](images/59344e4051b0864f617ad96a4a4983965218b3d369a18c575ec7e95b5f9f8439.jpg)

Causal forcing  
![](images/9be32d4fe73dade2916c3548418a1051a0c130b971a6269635409e22068acb69.jpg)  
A panda standing on a surfboard in the ocean in sunset.

Uncertainty DMD  
![](images/64d53f07d257d37a824d0cfd7c5827a9b5985cabf971db0a95c9f58d4db0c5bf.jpg)

![](images/a7e776915fe58505538881216bab712fe5b8c379d3425625c8f6137e575783c7.jpg)

![](images/b76b6d39b06650341651caaf7c2055e436f59287ab3bee9bb43f5cea96d5befa.jpg)  
A tree in an open field.

![](images/5b0d282902fbd20e1109e80071d894d84db3aadf0d5b6c77ea2e67a6123dc438.jpg)

![](images/44a1e35a3bf5f015d067e053b6daddcead7dc546e118ab7f2e79556d247e3ae7.jpg)  
5s 0s

![](images/9d4441e8d86757ab4cb1d5e8577a21627f052b729b5aa8445e032bcddc812012.jpg)  
A cat jumps on a low table.  
5s 0s

![](images/d0d14a30384579d91987ef145fadcbbc820df9161a1dda6efbad1ec5bc5588d7.jpg)  
Fig. 1. Comparison using 19 random seeds (0–18) under the same setup. The first two rows show the first frame of each generated video for cross-seed comparison. In the third row, each horizontal strip represents one video, with frames ordered from left to right.

Few-step distillation improves the eficiency of autoregressive (AR) video generation, but often causes diversity collapse: under the same prompt, different noise samples tend to produce highly similar videos with weakened motion dynamics. We analyze this degradation in Distribution Matching

Distillation (DMD)-distilled AR video generators and find that, in the autoregressive setting, it takes the form of a structured uncertainty collapse: the mode-seeking bias of DMD maps diferent noise samples to nearly identical first chunks, and the deterministic AR cache then propagates this collapsed state to all subsequent chunks, turning a local loss of stochasticity at the rollout root into a global suppression of temporal variation. Based on this analysis, we propose Uncertainty DMD, a simple uncertainty-injection framework that restores stochasticity at two key stages of AR generation: a timestep perturbation for the first chunk to increase first-chunk diversity, and a stochastic cache-writing mechanism for later chunks to preserve uncertainty in autoregressive conditioning. The method requires no architectural changes and introduces only lightweight perturbation operations. The same perturbation mechanisms are used during both training and inference. Ex periments show that Uncertainty DMD consistently improves diversity and motion dynamics while maintaining comparable per-sample visual quality. Project page: https://scdzx.github.io/Uncertainty-DMD

CCS Concepts: • Computing methodologies → Computer vision.

Additional Key Words and Phrases: Autoregressive Video Generation, Diversity, Distribution Matching Distillation, Uncertainty

## 1 Introduction

Recent advances in AI have enabled more capable models [An et al. 2026; Chen et al. 2026a; Gu et al. 2026; Huang et al. 2024a; Lu et al. 2025b; Shao and Li 2025; Wen et al. 2025; Zhang et al. 2026c]. Progress in image editing, audio-visual generation, and video compression has also advanced visual generative systems [Chen et al. 2026b, 2024a,b; Huang et al. 2025a; Song et al. 2026; Xiang and Fan 2025; Yuan et al. 2026a; Zang et al. 2026]. DMD is a distillation technique that transfers the generation capability of a multi-step diffusion teacher to a faster student requiring only a few denoising steps [Nguyen and Tran 2024; Shen et al. 2025; Yin et al. 2024a,b]. It has therefore been widely adopted for eficient image and video generation [Chen et al. 2025b; Geng et al. 2026; Lin et al. 2026; Sauer et al. 2024]. However, its distribution matching objective is modeseeking: the student may concentrate on a few high-density modes rather than covering the full teacher distribution, resulting in reduced sample diversity [Gandikota and Bau 2026; Luo et al. 2024; Wu et al. 2026]. This issue is particularly relevant to autoregressive video generation, where long videos are generated chunk by chunk using a history cache [Huang et al. 2025b; Yin et al. 2025; Yuan et al. 2026b]. To reduce the difusion sampling cost for each chunk, recent real-time autoregressive video generators often adopt DMD-based few-step distillation [Huang et al. 2025b; Zhu et al. 2026], and therefore inherit its mode-seeking tendency. Moreover, diversity collapse in early chunks can be stored in the cache and propagated to subsequent chunks, amplifying the loss of sample diversity throughout the video.

We observe this efect in DMD-distilled autoregressive video generators. As shown in Fig. 1, after DMD, both Self-Forcing [Huang et al. 2025b] and Causal Forcing [Zhu et al. 2026] tend to produce visually similar videos across diferent random noise under the same prompt. Although each individual video can still look plausible, the generated samples become much less diverse. In other words, few-step DMD preserves per-sample quality to a large extent, but substantially reduces per-prompt sample diversity and motion dynamics, which is an important property of the original multi-step generator.

![](images/c68ac92e13884f4da172353c4d1f804c6ad2fd4e79f90d538d261e94c26a8269.jpg)  
Fig. 2. Conceptual illustration of our distributional reshaping. While standard DMD distillation collapses the multi-modal ground-truth manifold into a deterministic single mode (middle), our Uncertainty Injection strategy preserves and expands mode coverage (right).

This loss of diversity has both an algorithmic and a structural origin, and is not merely image-style mode collapse. Algorithmically, DMD minimizes a reverse-KL-style objective against the teacher, which is known to be mode-seeking: the student concentrates probability mass on a few high-density modes rather than covering the full distribution [Liu et al. 2026a; Wu et al. 2026]. In AR video generation, diferent initial noises yield similar scenes, and motion is suppressed across time. Structurally, the autoregressive rollout amplifies both efects: the first chunk is generated without prior context and written into the history cache, which conditions all subsequent chunks and thereby fixes the scene layout, object placement, and coarse motion. Once diferent seeds collapse to similar first chunks, the resulting caches also collapse, while the per-step static bias further compresses temporal evolution. As conceptually illustrated in Fig. 2, this process progressively reduces mode coverage and drives the rollout toward a narrow deterministic trajectory. We call this failure mode uncertainty collapse: mode-seeking distillation eliminates stochasticity both at the root of the rollout (collapsing first chunks across noise) and along the temporal axis (collapsing motion within each video), and the deterministic cache propagates both forms of collapse forward, locking the rollout into a narrow set of near-static trajectories.

Motivated by this two-dimensional diagnosis, we propose Uncertainty DMD, a structured uncertainty-injection framework for few-step autoregressive video distillation. Instead of perturbing raw latents in an unstructured manner, Uncertainty DMD restores the lost uncertainty in few-step DMD by operating on two structured handles: the first-chunk timestep condition and the autoregressive cache that carries history into future chunks. The former diversifies the first chunk by perturbing its denoising timestep, while the latter preserves the resulting diversity across subsequent chunks by injecting stochasticity into the autoregressive cache. Notably, the same uncertainty injection is applied during both DMD training and inference, so that the student is optimized under the stochasticity it encounters at test time, avoiding the train–test mismatch of naive noise injection. This design recovers per-prompt diversity at negligible overhead and requires no architectural modification.

In summary, our contributions are threefold:

• Problem identification and analysis. We identify uncertainty collapse in DMD-distilled autoregressive video generators, where few-step distillation reduces diversity and suppresses motion. Through variance decomposition, cache transplantation, and teacher–DMD replacement studies, we show that uncertainty loss emerges in the first chunk and is further propagated by the autoregressive cache

• Uncertainty-injection framework. We propose Uncertainty DMD, which restores two structured sources of stochasticity suppressed by few-step DMD: the first-chunk timestep and the autoregressive cache. This creates and preserves diverse rollout branches with negligible overhead.

• Empirical validation. We validate Uncertainty DMD on two DMD-based AR video generators. Our analysis further reveals a trade-of in standard DMD distillation, where visual quality improves over training while diversity and motion dynamics progressively degrade. Uncertainty DMD substan tially mitigates this degradation, improving diversity and motion dynamics while preserving visual quality, all without modifying the original training and generation framework.

## 2 Related Work

Autoregressive Video Generation. While recent video difusion models achieve impressive visual quality, most rely on bidirectional or full-sequence denoising, which is costly and ill-suited for streaming or long-horizon synthesis [Blattmann et al. 2023; Hong et al. 2023; Hu 2024; Huang et al. 2026;Jia et al. 2026; Kong et al. 2024; Teng et al. 2025; Wan et al. 2025; Wang et al. 2026; Zhang et al. 2026a,b]. Autoregressive video generation instead synthesizes frames or chunks sequentially conditioned on previously generated content, enabling low-latency streaming and scalable long video synthesis [Chen et al. 2025a; Cui et al. 2026; Xiang et al. 2025, 2026a; Yin et al. 2025; Yuan et al. 2026b; Zhu et al. 2026]. To address the train–test mismatch and temporal drift inherent in causal generation, recent works explore self forcing training [Huang et al. 2025b], rolling windows, sink tokens, and explicit memory mechanisms [Huang et al. 2025b; Liu et al. 2026b; Sun et al. 2025; Xiang et al. 2026b; Yang et al. 2026; Yu et al. 2025], typically at the cost of architectural changes or additional retraining. Unlike these works that mainly target eficiency and temporal consistency, we focus on the intrinsic diversity collapse of distilled autoregressive video generators.

Few step Distillation and diversity collapse. Consistency Models [Song et al. 2023] preserve diversity better but achieve lower few-step quality. DMD accelerates difusion models by matching the output distribution of a few-step student to that of a multi step teacher [Fuest et al. 2026; Luo et al. 2023; Yin et al. 2024b; Zhou et al. 2025], and improved variants such as DMD2 [Yin et al. 2024a] achieve strong few-step generation quality. However, recent studies show that DMD-style objectives can suppress sample diversity due to their mode-seeking behavior. For image generation, f-DMD [Xu et al. 2025] introduces an additional discriminative score for �-divergence distillation, while ADM [Lu et al. 2025a] adopts GAN-based few-step distillation. DP-DMD [Wu et al. 2026] mitigates this issue in image generation by preserving diversity at the first distilled step with a target-prediction objective, while applying the DMD loss only to later refinement steps. Diversity Distillation [Gandikota and Bau 2026] observes that distilled models retain diversity-related representations but fail to activate them, and recovers diversity by skipping the first step or invoking an additional base model for the first denoising step, such first-step interventions may trade of sample quality. In video generation, DPP-GRPO [Kazimi et al. 2025] promotes video diversity by training a prompt-space policy to generate diverse prompt variants for a given input prompt, and evaluates diversity using metrics such as TCE [Ibarrola and Grace 2024], TIE [Ibarrola and Grace 2024], and VENDI [Friedman and Dieng 2023]. This prompt-level diversification is complementary to our fixed-prompt setting, where we restore diversity within the generation process itself. Diferent from these works, we study diversity collapse in DMD-based autoregressive video generation, where collapsed early chunks can propagate through the autoregressive cache and constrain future temporal evolution.

![](images/67f82b8e670f4cccd29d13adf055902c65159e3bc9aea153294024c90af9c539.jpg)  
Fig. 3. Cache Transplantation. Four videos generated by the same DMD student with identical noise seeds for all subsequent chunks; only the firstchunk cache difers. Top: first-chunk cache from a multi-step teacher (Multistep Cached). Botom: first-chunk cache from the DMD student itself (DMD Cached). Prompt: “a cat wearing sunglasses in the pool”.

## 3 Diagnosing Mode Collapse in Few-Step AR Distillation

## 3.1 Few-Step AR Video Generation and Diversity Evaluation

AR video generation represents a video as a sequence of temporal chunks $x _ { 1 : L } ~ = ~ \{ x _ { 1 } , \ldots , x _ { L } \}$ and generates them according to the chunk-wise [Huang et al. 2025b; Yang et al. 2026] conditional

factorization

$$
p _ { \theta } ( x _ { 1 : L } \mid c ) = p _ { \theta } ( x _ { 1 } \mid c ) \prod _ { i = 2 } ^ { L } p _ { \theta } ( x _ { i } \mid x _ { < i } , c ) ,\tag{1}
$$

where � denotes the text condition.

In practical AR video models, previously generated chunks are stored in an autoregressive cache $C _ { i - 1 } . \operatorname { A } K$ -step distilled student generates the �-th chunk as

$$
\begin{array} { r } { \hat { x } _ { i , 0 } = G _ { \theta } ^ { ( K ) } \big ( z _ { i } , t _ { 1 : K } , C _ { i - 1 } , c ; \epsilon _ { 1 : K } \big ) , } \end{array}\tag{2}
$$

where $z _ { i }$ is the initial noise, $t _ { 1 : K }$ is the few-step timestep schedule, and $\epsilon _ { 1 : K }$ denotes the stochasticity during sampling. After each chunk is generated, the cache is updated by a deterministic write operation:

$$
\begin{array} { r } { C _ { i } = \mathrm { A p p e n d } \left( C _ { i - 1 } , \hat { x } _ { i , 0 } \right) . } \end{array}\tag{3}
$$

To evaluate generation diversity under the same prompt, we generate multiple videos with diferent random seeds, extract videolevel features, and compute truncated entropy based on the covari ance spectrum of these features [Ibarrola and Grace 2024] and the spectrum ofthe sample-similarity kernel [Friedman and Dieng 2023]. Higher values indicate that the generated videos are more dispersed in feature space and therefore exhibit stronger diversity. Following DPP-GRPO [Kazimi et al. 2025], we adopt the same computation protocol and extend it to six diversity metrics; detailed definitions and implementation details are provided in Supplementary.

## 3.2 DMD Mode Collapse and First Chunk Collapse

Few-step students are often trained with DMD to match a multi-step teacher [Yin et al. 2024a,b]. Given a generated sample $x = G _ { \theta } ( z , c )$ and its noisy version $x _ { t } .$ , the DMD gradient can be written as

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { D M D } } = \mathbb { E } _ { z , t } \left[ w ( t ) \left( s _ { \mathrm { f a k e } } ( x _ { t } , t , c ) - s _ { \mathrm { r e a l } } ( x _ { t } , t , c ) \right) \frac { \partial G _ { \theta } ( z , c ) } { \partial \theta } \right] ,\tag{4}
$$

where $s _ { \mathrm { r e a l } }$ is the frozen teacher score and $S f a k e$ is estimated by an auxiliary fake-score model. Following [Huang et al. 2025b], $w ( t ) = 1 / \mathrm { m e a n } ( | x - s _ { \mathrm { r e a l } } ( x _ { t } , t , c ) | )$ normalizes the gradient magnitude. Similar to the model-collapse phenomenon observed in imagelevel DMD, this score-diference update tends to encourage the student to concentrate on high-density teacher regions, which can reduce sample diversity under diferent random noise [Gandikota and Bau 2026; Luo et al. 2024; Wu et al. 2026].

This efect is particularly visible in the first chunk of AR video generation. The first chunk has no historical cache condition,

$$
C _ { 0 } = \varnothing ,\tag{5}
$$

so global factors such as scene layout, object pose, camera viewpoint, and initial motion must be determined only from the prompt and random noise:

$$
\hat { x } _ { 1 , 0 } = G _ { \theta } ^ { ( K ) } \left( z _ { 1 } , t _ { 1 : K } , C _ { 0 } , c ; \epsilon _ { 1 : K } \right) .\tag{6}
$$

Empirically, under the mode-seeking bias offew-step DMD, diferent seeds may be mapped to similar high-probability first chunks,

$$
G _ { \theta } ^ { ( K ) } \left( z _ { 1 } ^ { a } , t _ { 1 : K } , C _ { 0 } , c \right) \approx G _ { \theta } ^ { ( K ) } \left( z _ { 1 } ^ { b } , t _ { 1 : K } , C _ { 0 } , c \right) , \qquad z _ { 1 } ^ { a } \neq z _ { 1 } ^ { b } .\tag{7}
$$

We call this phenomenon first-chunk collapse: for the same prompt, diferent noise already produce similar layouts, poses, viewpoints, or initial motion patterns in the first generated chunk.

![](images/f11b1bb602cae096c498a28a3cf790e2e8cc34ae3d310a4a8e7d94b0b767ae09.jpg)  
Fig. 4. Empirical variance decomposition of later-chunk diversity in DMDbased AR video generation. For each prompt and each history length, we sample 8 diferent history caches and 8 continuations for each cache.

## 3.3 Deterministic Cache Propagates Diversity Collapse

For later chunks, the student generates under the historical cache condition:

$$
\hat { x } _ { i , 0 } = G _ { \theta } ^ { ( K ) } \bigl ( z _ { i } , t _ { 1 : K } , C _ { i - 1 } , c ; \epsilon _ { 1 : K } \bigr ) , \qquad i > 1 .\tag{8}
$$

Thus, the diversity of a later chunk can be decomposed into two parts: the variation produced by the current few-step sampler under a fixed cache, and the variation caused by diferent historical caches. Let �(·) denote the feature of a generated chunk. By total variance decomposition,

$$
\begin{array} { r l } & { \mathrm { V a r } \big [ \phi ( \hat { x } _ { i , 0 } ) \mid c \big ] = \mathbb { E } _ { C _ { i - 1 } } \left[ \mathrm { V a r } \big [ \phi ( \hat { x } _ { i , 0 } ) \mid C _ { i - 1 } , c \big ] \right] } \\ & { ~ + \mathrm { V a r } _ { C _ { i - 1 } } \left[ \mathbb { E } \big [ \phi ( \hat { x } _ { i , 0 } ) \mid C _ { i - 1 } , c \big ] \right] . } \end{array}\tag{9}
$$

The first term is the innovation diversity, which measures how much new semantic or motion variation the current sampler can still introduce when the history cache is fixed. The second term is the cache-inherited diversity, which measures how much diferent cache states lead to diferent future continuations.

We further estimate the two terms in Eq. (9) on DMD-based autoregressive video generation. We use 100 prompts and vary the history chunk length $h \in \{ 1 , 2 , 3 , 4 \}$ , where each history chunk contains 3 latent frames and each rollout is fixed to 7 chunks. For each prompt and each history length, we first generate 8 diferent history caches $\{ C _ { h } ^ { a } \} _ { a = 1 } ^ { 8 }$ , and for each cache we generate 8 diferent next-chunk continuations using diferent continuation noise. This gives 8 × 8 × 4 generated videos for each prompt. The variance under the same cache estimates the innovation term, while the variance across diferent cache means estimates the cache-inherited term.

As shown in Fig. 4, the cache-inherited term dominates the total variance for all history lengths. It accounts for 0.94, 0.96, 0.97, and 0.97 of the total variance when the history length increases from 1 to 4, while the innovation term only contributes a small fraction. This result indicates that, in few-step DMD autoregressive generation, the historical cache is the main source controlling the diversity of later chunks. Once diferent noise are mapped to similar cache states, the later rollout has limited ability to recover diversity only through the current sampler.

Few-step DMD first reduces first-chunk diversity, as the sampler becomes less sensitive to random noise. Once similar first chunks are written into the AR cache, later few-step chunks are conditioned on similar histories and continue to inherit the collapsed state. Since the cache update is deterministic,

Table 1. Diagnosing diversity collapse on Causal Forcing. Sampler choice for the first vs. later chunks isolates where diversity is lost.
<table><tr><td></td><td colspan="3">Truncated Entropy ↑</td><td colspan="3">VENDI↑</td></tr><tr><td>Configuration</td><td>CLIP</td><td>INC</td><td>DINO</td><td>CLIP</td><td>INC</td><td>DINO</td></tr><tr><td>Teacher / Teacher</td><td>22.1</td><td>39.4</td><td>72.5</td><td>2.3</td><td>3.5</td><td>4.9</td></tr><tr><td>Teacher / DMD</td><td>21.4</td><td>39.1</td><td>71.8</td><td>2.2</td><td>3.4</td><td>4.6</td></tr><tr><td>DMD / Teacher</td><td>19.2</td><td>37.1</td><td>69.2</td><td>2.0</td><td>3.1</td><td>3.6</td></tr><tr><td>DMD / DMD</td><td>19.8</td><td>37.4</td><td>69.3</td><td>2.0</td><td>3.0</td><td>3.6</td></tr></table>

$$
\begin{array} { r } { C _ { i } = \mathrm { A p p e n d } \left( C _ { i - 1 } , \hat { x } _ { i , 0 } \right) , } \end{array}\tag{10}
$$

the cache cannot create new high-level semantic branches by itself; it mainly propagates the diversity, or lack of diversity, already present in previous chunks.

Fig. 3 illustrates this efect through cache transplantation. We use the same DMD student and keep identical noise for all subsequent chunks, changing only the first-chunk cache. When the cache comes from the multi-step teacher, later chunks inherit more diverse historical conditions. In contrast, when the cache comes from the DMD student itself, the subsequent generations become much more similar. This shows that diversity collapse in few-step AR distillation is not only a local sampling issue, but is propagated along the rollout by the deterministic AR cache.

## 3.4 Testable Prediction and Motivation

The above analysis gives a direct testable prediction. If diversity collapse mainly originates from the first chunk and is then propagated by the AR cache, then replacing only the first-chunk sampler should recover most of the lost diversity. In contrast, replacing only the later-chunk sampler should provide limited improvement, because later chunks are still conditioned on a collapsed first cache.

Let � denote the multi-step teacher sampler and � denote the few-step DMD student. We expect

$$
\operatorname { D i v } ( T _ { 1 } , Q _ { > 1 } ) - \operatorname { D i v } ( Q _ { 1 : L } ) \gg \operatorname { D i v } ( Q _ { 1 } , T _ { > 1 } ) - \operatorname { D i v } ( Q _ { 1 : L } ) ,\tag{11}
$$

where Div(·) denotes a cross-sample diversity metric, such as Truncated Entropy [Ibarrola and Grace 2024] or VENDI [Friedman and Dieng 2023]. Here, $( T _ { 1 } , Q _ { > 1 } )$ means using the teacher only for the first chunk and the DMD student for later chunks, while $( Q _ { 1 } , T _ { > 1 } )$ means using the DMD student for the first chunk and the teacher for later chunks.

Table 1 validates this prediction. Replacing only the first chunk with the teacher sampler (Teacher / DMD) recovers most of the diversity compared with the full DMD rollout (DMD / DMD). In contrast, replacing only later chunks (DMD / Teacher) brings much smaller gains, showing that the collapsed first-chunk cache already constrains the subsequent rollout.

Taken together, the three analyses above reveal that diversity collapse in few-step AR distillation is not only the known reverse-KL-like, mode-seeking behavior of DMD observed in image distillation [Gandikota and Bau 2026; Wu et al. 2026], but is further amplified by the autoregressive structure. The first-chunk collapse experiment (Sec. 3.2) shows that few-step DMD maps diferent noise to nearly identical first chunks; the variance decomposition (Fig. 4)

shows that, once the cache is fixed, the sampler contributes little innovation diversity, with more than 94% of later-chunk variance inherited from the cache; and the cache transplantation and samplerswap studies (Fig. 3, Table 1) confirm that the collapsed first chunk is written into the deterministic history cache and then constrains all subsequent chunks. Therefore, perturbing intermediate latents in later denoising steps has limited ability to recover missing modes. Instead, uncertainty should be injected into the two AR state variables that actually steer the rollout: the first-chunk denoising trajectory and the autoregressive cache writing. This directly motivates our Uncertainty DMD.

## 4 Methodology

Uncertainty DMD injects stochasticity into the two AR state variables: the first-chunk timestep condition and the cache written for future chunks. Sec. 4.1 introduces timestep perturbation, Sec. 4.2 introduces stochastic cache writing, and Sec. 4.3 gives the curriculum schedule and overall algorithm. The overall framework of Uncertainty DMD is illustrated in Fig. 6.

## 4.1 Module A: Timestep Perturbation

As described in Sec. 3.1, the student generates each video chunk with a few-step denoising schedule $\tau _ { 1 } = T > \tau _ { 2 } > \cdot \cdot \cdot > \tau _ { K } .$ . In few-step distilled samplers, the first denoising call largely fixes the coarse layout and motion, while later calls mainly refine it. Because this call starts from almost pure noise, a fixed timestep makes every sample take the same large denoising update, which can wash out noise diferences and push outputs toward similar modes. We perturb this timestep condition so the first update is made under slightly diferent noise levels, allowing diferent noise to choose diferent early modes. This also explains why skipping the first step in DMD image models [Gandikota and Bau 2026] can increase diversity: feeding a noisy latent with a randomly sampled cleaner timestep $T - \Delta t$ causes part of the unresolved noise to be interpreted as meaningful structural variation, leading to more diverse trajectories, while skipping the first step similarly avoids the mode-collapsing initial update.

The first chunk starts from the initial noisy latent $x _ { 1 , T } ,$ . When first timestep perturbation is activated, we keep this latent unchanged but replace the timestep condition � by a perturbed timestep

$$
\tilde { T } = T - \Delta , \qquad \Delta \sim p _ { \Delta } ,
$$

where $\tilde { T }$ is clipped to the valid timestep range, and $\mathcal { P } \Delta$ is the timestepofset distribution, implemented as $\Delta \sim \mathcal { U } ( 0 , 1 0 0 )$ . The first generator call for the first chunk becomes

$$
\hat { x } _ { 1 , 0 } ^ { ( 1 ) } = G _ { \theta } ( x _ { 1 , T } , \tilde { T } , C _ { 0 } , c ) , \qquad C _ { 0 } = \emptyset .
$$

Equivalently, the generator receives the same latent as in the vanilla few-step sampler, but is conditioned on a slightly shifted noise level $T - \Delta$ . This timestep mismatch changes the first-chunk denoising trajectory without directly perturbing the latent itself, thereby inducing diferent branches for the subsequent autoregressive rollout.

During training, this operation is applied to the generator rollout with probability $p _ { \mathrm { r o o t } } ( s )$ , which is gradually increased by the curriculum schedule in Sec. 4.3.

![](images/ac0573f8f64959bdc41b7f390e0e42e7090874a499b86fdf7099432575030f13.jpg)  
Fig. 5. Overview of the proposed Uncertainty Injection DMD framework. Compared with standard DMD, our method perturbs the structured autoregressive state by injecting timestep uncertainty into the first chunk and stochastic noise into cache writing.

## 4.2 Module B: Stochastic Cache Writing

For later chunks, we inject uncertainty into the autoregressive conditioning through a stochastic cache-writing mechanism. Under self-forcing training, the AR cache is constructed from the model’s own rollout. At autoregressive step $j ,$ the rollout reaches a sampled truncation step and the model outputs a clean prediction $\hat { x } _ { j , 0 }$ for the current chunk. Before writing this predicted clean chunk into the cache, we optionally perturb it once. After insertion, the cached representation is kept fixed and is not re-perturbed at later autoregressive steps.

We define the forward noising operator as

$$
\Psi ( x , \epsilon , \tau ) = \alpha _ { \tau } x + \sigma _ { \tau } \epsilon ,\tag{12}
$$

where $\alpha _ { \tau }$ and $\sigma _ { \tau }$ are predefined coeficients associated with noise level �.

Before cache insertion, the predicted clean chunk is perturbed as

$$
\bar { x } _ { j , 0 } = \big ( 1 - m _ { j } \big ) \hat { x } _ { j , 0 } + m _ { j } \Psi \big ( \hat { x } _ { j , 0 } , \epsilon _ { j } , \tau _ { \mathrm { c a c h e } } \big ) ,\tag{13}
$$

where

$$
m _ { j } \sim \mathrm { B e r n o u l l i } ( \phi _ { \mathrm { c a c h e } } ) , \qquad \epsilon _ { j } \sim N ( 0 , { \bf I } ) .\tag{14}
$$

Here, $\mathcal { P } \mathrm { c a c h e }$ denotes the probability ofperturbing the predicted clean chunk before cache insertion, and $\tau _ { \mathrm { c a c h e } }$ controls the perturbation strength through $\alpha _ { \tau _ { \mathrm { c a c h e } } }$ and $\sigma _ { \tau _ { \mathrm { c a c h e } } }$

The perturbed chunk is then written into the AR cache:

$$
\tilde { C } _ { j } = \mathrm { A p p e n d } \left( \tilde { C } _ { j - 1 } , \bar { x } _ { j , 0 } \right) , \qquad \tilde { C } _ { 0 } = \emptyset .\tag{15}
$$

Thus, when generating the �-th chunk, the cache $\tilde { C } _ { i - 1 }$ contains fixed entries $\{ \bar { x } _ { 1 , 0 } , \bar { x } _ { 2 , 0 } , \ldots , \bar { x } _ { i - 1 , 0 } \}$ , each sampled and written only once during the rollout. The student predicts the current chunk conditioned on this perturbed AR cache:

$$
\hat { x } _ { i , 0 } = G _ { \theta } ( x _ { i , t } , t , \tilde { C } _ { i - 1 } , c ) , \qquad i > 1 .\tag{16}
$$

The same stochastic cache-writing rule is used in both training and inference. During training, the cache is recursively constructed from previously generated clean predictions along the self-forcing rollout. During inference, each newly generated chunk is optionally perturbed once before cache insertion and then kept fixed for all subsequent autoregressive steps.

## 4.3 Uncertainty Curriculum and Overall Algorithm

We apply the two structured perturbations with a simple curriculum schedule. At the early stage of DMD training, both perturbations are disabled. After a warm-up iteration $S _ { w } ,$ their probabilities are gradually increased with the training iteration. Let ℓ denote the current training iteration. We define

$$
\gamma _ { \ell } = \mathrm { c l i p } \left( \frac { \ell - S _ { w } } { R } , 0 , 1 \right) ,
$$

where � is the ramp-up length. The probabilities of root timestep perturbation and stochastic cache writing are

$$
\begin{array} { r } { p _ { \mathrm { r o o t } } ( \ell ) = \gamma _ { \ell } , \qquad p _ { \mathrm { c a c h e } } ( \ell ) = \gamma _ { \ell } p _ { \mathrm { c a c h e } } ^ { \mathrm { m a x } } . } \end{array}
$$

Thus, after the curriculum finishes, timestep perturbation is always enabled, while cache writing perturbation is applied with a high probability $p _ { \mathrm { c a c h e } } ^ { \mathrm { m a x } }$ . At inference, we directly use the final probabilities, i.e. $, p _ { \mathrm { r o o t } } = 1 \widetilde { \mathrm { a n d } } p _ { \mathrm { c a c h e } } = p _ { \mathrm { c a c h e } } ^ { \mathrm { m a x } }$ . The overall training procedure with the proposed structured uncertainty is summarized in Alg. 1.

## 5 Experiments

Implementation. We implement our approach on two AR video generation frameworks, Causal Forcing [Zhu et al. 2026] and Self-Forcing [Huang et al. 2025b]. Both are built upon the Wan2.1-T2V-1.3B [Wan et al. 2025] to generate 5-second videos at 16 FPS with a resolution of 832 × 480. Following baseline protocols, we adopt a chunk-wise AR scheme generating a chunk of 3 latent frames at a time [Yin et al. 2025], and utilize text prompts from a Vid-ProM [Wang and Yang 2024] dataset for data-free DMD. All models are trained on 32 H100 GPUs with a batch size of 32. Unless otherwise specified, the default training duration is set to 600 steps. Additional experimental details, including dataset construction, training hyperparameters, inference settings, and metric computation, are provided in Supplementary.

Algorithm 1 DMD Training with Structured Uncertainty   
Require: AR generator $G _ { \theta } ,$ KV encoder $G _ { \theta } ^ { \mathrm { K V } } :$ , warm-up $S _ { w } ,$ ramp   
length �, maximum cache probability $p _ { \mathrm { c a c h e } } ^ { \mathrm { \bar { m a x } } }$   
1: for training iteration $\ell = 1 , 2 , \ldots$ do   
2: Compute curriculum strength $\gamma _ { \ell } = \mathrm { c l i p } ( ( \ell - S _ { w } ) / R , 0 , 1 ) .$   
3: Set $p _ { \mathrm { r o o t } } = \gamma _ { \ell }$ and $p _ { \mathrm { c a c h e } } = \gamma _ { \ell } p _ { \mathrm { c a c h e } } ^ { \mathrm { m a x } } .$   
4: Sample training chunks and initialize AR cache $\mathrm { K V } \gets [ ] .$   
5: for chunk index $i = 1 , \ldots , K$ do   
6: Sample $u \sim \mathcal { U } ( 0 , 1 ) .$   
7: if $i = 1$ and $u < p _ { \mathrm { r o o t } }$ then   
8: Replace the first denoising timestep   
$\tau _ { 1 } = T  \tilde { \tau } _ { 1 } = T - \Delta , \Delta \sim p _ { \Delta }$ {root timestep   
perturbation}   
9: end if   
10: Generate chunk $\hat { x } _ { 0 } ^ { i }$ by the standard DMD rollout with the   
current schedule and cache KV.   
11: Set $\hat { x } _ { 0 } ^ { i }  \hat { x } _ { 0 } ^ { i } .$   
12: Sample $\begin{array} { r } { v \sim \mathcal { U } ( 0 , 1 ) . } \end{array}$   
13: $\mathbf { i f } ~ v < p _ { \mathrm { c a c h e } }$ then   
14: $\bar { x } _ { 0 } ^ { i } \gets \Psi ( \hat { x } _ { 0 } ^ { i } , \epsilon _ { i } , \tau _ { \mathrm { c a c h e } } ) , \epsilon _ { i } \sim N ( 0 , \mathrm { I } )$ {stochastic cache   
writing}   
15: end if   
16: Write cache: $\mathrm { K V }  \mathrm { K V } \cup G _ { \theta } ^ { \mathrm { K V } } ( \bar { x } _ { 0 } ^ { i } , 0 , \mathrm { K V } )$   
17: end for   
18: Update $G _ { \theta }$ with the standard DMD objective.   
19: end for   
20: Inference: use $\mathcal { P } _ { \mathrm { r o o t } } = 1$ and $\ p _ { \mathrm { c a c h e } } =  { p _ { \mathrm { c a c h e } } } ^ { \mathrm { m a x } }$ with the same per  
turbation rules.

Evaluation. We use VBench [Huang et al. 2024b] as our primary benchmark to evaluate both intrinsic diversity and generation quality. We randomly sample 100 text prompts from the VBench suite and generate 20 videos per prompt with a fixed set of random seeds shared across all methods, resulting in 2,000 videos for each method. Following DPP-GRPO [Kazimi et al. 2025], we measure set-level diversity among videos generated from the same prompt. We extend the three diversity metrics in DPP-GRPO to six metrics by computing both Truncated Entropy [Ibarrola and Grace 2024] and VENDI scores [Friedman and Dieng 2023] on three feature spaces: CLIP [Radford et al. 2021], Inception [Szegedy et al. 2016], and DINO [Oquab et al. 2024]. These metrics are denoted as TE-CLIP, TE-INCEPTION, TE-DINO, VENDI-CLIP, VENDI-INCEPTION, and VENDI-DINO. We further report the standard VBench quality metrics over all 2,000 generated videos.

Quantitative results. Table 2 reports a quantitative comparison of video diversity and generation quality. Our method consistently improves video diversity while maintaining comparable VBench quality. This shows that the proposed structured uncertainty efectively recovers noise-dependent variation in autoregressive video distillation. Restored diversity recovers more motion modes, allowing more samples to pass the dynamic threshold underlying Dynamic Degree, a high-variance binary metric; DMD + GAN shows a similar trend. Moreover, autoregressive models can achieve higher Dynamic Degree than bidirectional baselines, as also observed in Causal Forcing. We also compare with two auxiliary baselines under the Self-Forcing initialization. DMD + GAN only brings limited diversity improvement, and DMD + Extra Noise does not provide consistent gains. Here, DMD + Extra Noise directly adds Gaussian noise to the generator latent during training:

$$
\tilde { x } _ { i , t } = x _ { i , t } + m _ { i } \cdot 0 . 1 \epsilon _ { i } , \quad m _ { i } \sim \mathrm { B e r n o u l l i } ( 0 . 1 ) , \ \epsilon _ { i } \sim N ( 0 , I ) .
$$

In contrast, our method perturbs structured AR variables, namely the timestep and the historical cache, leading to more efective diversity recovery. In Supplementary, we further show that our method outperforms two alternatives, i.e., skipping the first denoising step in the first chunk and replacing it with the multi-step teacher sampler, yielding better diversity recovery with comparable quality.We also conduct a user study, where our method receives comparable perceptual quality scores to standard DMD while obtaining substantially higher diversity scores.

Ablation study. Table 3 ablates the key components of our method under the Self-Forcing initialization. Removing timestep perturbation significantly reduces diversity, e.g., TE-CLIP drops from 19.5 to 16.7 and VENDI-DINO drops from 3.3 to 2.6, showing that injecting uncertainty into the initial chunk is crucial for preserving seeddependent variations. Removing stochastic cache writing keeps most diversity metrics close to the full model, but lowers the dynamic degree from 42.4 to 37.0, indicating that perturbing the historical cache mainly helps maintain stronger temporal dynamics. Without the curriculum schedule, the model still improves over vanilla DMD, but both diversity and motion dynamics degrade compared with the full method: TE-CLIP decreases from 19.5 to 18.4, VENDI-DINO decreases from 3.3 to 3.1, and the dynamic degree drops from 42.4 to 37.4. This suggests that gradually introducing the proposed uncertainty perturbations stabilizes training and helps the model better adapt to stochastic initial states and noisy historical caches. Overall, the full method achieves the best balance between diversity and dynamic generation quality.

Generalization study. Following Causal Forcing [Zhu et al. 2026], we extend our method to DMD with the Wan2.1 1.3B frame-wise and 14B chunk-wise models. As shown in Table 4, the results show its generalizability across diferent model sizes and temporal granularities.

Training dynamics. Figure 6 compares DMD, DMD + GAN, and DMD + Ours at diferent training steps. All methods show similar quality trends during training, while our method maintains competitive VBench quality and a higher dynamic degree in later stages. For diversity, DMD + Ours consistently outperforms DMD and DMD + GAN across training steps, indicating that the proposed uncertainty injection slows down diversity degradation during distillation.

Timestep perturbation. Figure 7 shows the first-step denoising behavior of the few-step model with fixed initial noise and text prompt, while varying only the input timestep. The predictions vary noticeably, indicating that timestep perturbation introduces additional stochasticity and enables diverse generation trajectories from the same noise.

Table 2. Quantitative comparison of video diversity and generation quality on the 2,000 generated videos
<table><tr><td rowspan="2">Method</td><td colspan="6">Video Diversity (↑)</td><td colspan="6">Generation Quality (VBench) (↑)</td></tr><tr><td>TE-CLIP</td><td>TE-Inc.</td><td>TE-DINO</td><td>VENDI-CLIP</td><td>VENDI-Inc.</td><td>VENDI-DINO</td><td>Sub. Cons.</td><td>Bg. Cons.</td><td>Motion Sm.</td><td>Dyn. Deg.</td><td>Aesthetic</td><td>Imaging</td></tr><tr><td>Causal Forcing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DMD</td><td>19.8</td><td>37.4</td><td>69.3</td><td>2.0</td><td>3.0</td><td>3.6</td><td>98.1</td><td>97.4</td><td>99.1</td><td>31.1</td><td>67.4</td><td>73.3</td></tr><tr><td>DMD + Ours</td><td>22.1</td><td>39.4</td><td>71.9</td><td>2.3</td><td>3.5</td><td>4.7</td><td>97.3</td><td>97.0</td><td>99.0</td><td>41.8</td><td>66.6</td><td>73.5</td></tr><tr><td>Self Forcing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DMD</td><td>16.4</td><td>33.5</td><td>66.1</td><td>1.6</td><td>2.4</td><td>2.6</td><td>98.6</td><td>97.6</td><td>99.2</td><td>25.1</td><td>65.9</td><td>72.5</td></tr><tr><td>DMD + GAN</td><td>16.8</td><td>33.7</td><td>66.3</td><td>1.7</td><td>2.4</td><td>2.6</td><td>97.8</td><td>96.8</td><td>98.9</td><td>40.4</td><td>65.9</td><td>72.7</td></tr><tr><td>DMD + Extra Noise</td><td>16.1</td><td>33.2</td><td>65.6</td><td>1.7</td><td>2.4</td><td>2.5</td><td>97.9</td><td>97.9</td><td>99.3</td><td>15.9</td><td>66.5</td><td>73.2</td></tr><tr><td>DMD + Ours</td><td>19.5</td><td>36.1</td><td>68.6</td><td>1.9</td><td>2.8</td><td>3.3</td><td>97.6</td><td>96.6</td><td>98.7</td><td>42.4</td><td>64.6</td><td>72.8</td></tr><tr><td>Wan (Bi-directional)</td><td>20.9</td><td>38.0</td><td>71.8</td><td>2.0</td><td>3.3</td><td>4.5</td><td>98.2</td><td>98.4</td><td>99.1</td><td>22.7</td><td>65.7</td><td>71.3</td></tr></table>

![](images/5b01b0264880871e68e24dab5dbb5a92fe7c8394985f32d3e920ab1eccc54881.jpg)  
Fig. 6. Quality and diversity metrics across training steps.

Table 3. Ablation study of the proposed components.
<table><tr><td colspan="7">(a) Video Diversity (↑)</td></tr><tr><td>Method</td><td>VENDI-CLIP</td><td>VENDI-Inc.</td><td>VENDI-DINO</td><td>TE-CLIP</td><td>TE-Inc.</td><td>TE-DINO</td></tr><tr><td>Ours</td><td>1.9</td><td>2.8</td><td>3.3</td><td>19.5</td><td>36.1</td><td>68.6</td></tr><tr><td>w/o Timestep Perturbation</td><td>1.7</td><td>2.5</td><td>2.6</td><td>16.7</td><td>33.3</td><td>65.7</td></tr><tr><td>w/o Stochastic Cache Writing</td><td>1.8</td><td>2.8</td><td>3.2</td><td>19.1</td><td>36.0</td><td>68.5</td></tr><tr><td>w/o Curriculum</td><td>1.8</td><td>2.8</td><td>3.1</td><td>18.4</td><td>35.5</td><td>68.0</td></tr><tr><td colspan="7">(b) Generation Quality (VBench) (↑)</td></tr><tr><td>Method</td><td>Sub. Cons.</td><td>Bg. Cons.</td><td>Motion Sm.</td><td>Dyn. Deg.</td><td>Aesthetic</td><td>Imaging</td></tr><tr><td>Ours</td><td>97.6</td><td>96.6</td><td>98.7</td><td>42.4</td><td>64.6</td><td>72.8</td></tr><tr><td>w/o Timestep Perturbation</td><td>98.3</td><td>97.3</td><td>98.9</td><td>32.6</td><td>66.3</td><td>72.7</td></tr><tr><td>w/o Stochastic Cache Writing</td><td>98.4</td><td>97.3</td><td>98.9</td><td>37.0</td><td>65.3</td><td>73.0</td></tr><tr><td>w/o Curriculum</td><td>97.6</td><td>96.8</td><td>98.8</td><td>37.4</td><td>64.9</td><td>72.5</td></tr></table>

Table 4. Comparison of diversity and quality under diferent model setings.
<table><tr><td>Method</td><td>VENDI-CLIP</td><td>VENDI-Inc.</td><td>VENDI-DINO</td><td>TE-CLIP</td><td>TE-Inc.</td><td>TE-DINO</td></tr><tr><td>Frame-wise 1.3B + DMD</td><td>1.9</td><td>2.8</td><td>3.2</td><td>18.4</td><td>35.4</td><td>67.8</td></tr><tr><td>Frame-wise 1.3B + Ours</td><td>2.2</td><td>3.3</td><td>4.2</td><td>21.5</td><td>38.2</td><td>71.0</td></tr><tr><td>Chunk-wise 14B + DMD</td><td>1.8</td><td>2.7</td><td>3.0</td><td>17.3</td><td>34.9</td><td>67.2</td></tr><tr><td>Chunk-wise 14B + Ours</td><td>2.2</td><td>3.4</td><td>4.4</td><td>21.8</td><td>39.0</td><td>71.5</td></tr><tr><td colspan="7">(b) Generation Quality (VBench) (↑)</td></tr><tr><td>Method</td><td>Sub. Cons.</td><td>Bg. Cons.</td><td>Motion Sm.</td><td>Dyn. Deg.</td><td>Aesthetic</td><td>Imaging</td></tr><tr><td>Frame-wise 1.3B + DMD</td><td>98.3</td><td>97.0</td><td>99.3</td><td>21.1</td><td>65.7</td><td>72.6</td></tr><tr><td>Frame-wise 1.3B + Ours</td><td>97.0</td><td>96.1</td><td>99.1</td><td>38.3</td><td>64.6</td><td>72.3</td></tr><tr><td>Chunk-wise 14B + DMD</td><td>98.7</td><td>98.1</td><td>99.4</td><td>32.9</td><td>66.6</td><td>72.3</td></tr><tr><td>Chunk-wise 14B + Ours</td><td>98.3</td><td>97.1</td><td>99.1</td><td>36.6</td><td>64.6</td><td>72.5</td></tr></table>

## 6 Conclusion

We propose uncertainty DMD to mitigate diversity collapse in DMD-based autoregressive video difusion distillation. By combining timestep perturbation with stochastic cache writing, our method improves sample diversity and motion dynamics while maintaining comparable visual quality. Nevertheless, current autoregressive video generation lacks explicit planning for future motion, which may result in unnatural complex motions.

![](images/b5d037d6f019dcf7e7219b9c50d0c73a8421cfa0506e9a19490db5501ce00af4.jpg)  
Fig. 7. Visualization of noise and timestep efects on the first denoising step of the first frame using four-step Self-Forcing model. All results use the same prompt, “a beautiful bird.” Noise: diferent random noise samples at timestep 1000. Timestep: four evenly spaced timesteps from 1000 to 950 (left to right), using the same noise sample.

## Acknowledgments

This work was supported in part by the Natinal Natural Science Foundation of China, under Grant Numbers 62192783, 62276128, and 62406140; the Young Elite Scientists Sponsorship Program by China Association for Science and Technology, under Grant Number 2023QNRC001; the Key Research and Development Program of Jiangsu Province, under Grant Number BE2023019; and the Jiangsu Natural Science Foundation, under Grant Nos. BK20221441 and BK20241200. The authors would like to thank the support of Huawei Ascend Cloud Ecological Development Project.

![](images/09e274e7cf5e976a22886ccc223c63c1754e968b5d2fe32356794f7378ac54c1.jpg)  
Fig. 8. Diversity comparison results. All methods are initialized from the same Self-Forcing ODE model. For a fair comparison, we keep the random seeds identical across methods within each row, and generate 19 videos using seeds from 0 to 18 for each prompt. The prompts are as follows: (a) An astronaut flying in space, featuring a steady and smooth perspective; (b) Grass swaying in the breeze; (c) A tropical beach at sunrise, with palm trees and crystal-clear water in the foreground; (d) A statue in a garden; (e) A 3D model of a 1800s victorian house.; (f ) A person is walking through a night market.

DMD

DMD + GAN

DMD + Ours

![](images/635713a3fe6248c01fbbfe0b46e8cf0b98cb8de8bf7af82939f35d02bdcff144.jpg)  
Fig. 9. Diversity and temporal comparison results. All methods are initialized from the same Self-Forcing ODE model. For a fair comparison, we keep the random seeds identical across methods. The prompt of each case is shown above the corresponding group. For each prompt, the left side presents four videos generated with diferent seeds, and the right side shows the temporal evolution of these videos by uniformly sampling four frames from each corresponding 81-frame sequence.

## References

Hongjun An, Wenhan Hu, Sida Huang, Siqi Huang, Ruanjun Li, Yuanzhi Liang, Jiawei Shao, Yiliang Song, Zihan Wang, Cheng Yuan, et al. 2026. Ai flow: Perspectives, scenarios, and approaches. Vicinagearth 3, 1 (2026), 1.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. 2023. Align your latents: High-resolution video synthesis with latent difusion models. In CVPR. 22563–22575.

Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. 2025b. Sana-sprint: One-step difusion with continuous-time consistency distillation. In ICCV. 16185–16195.

Tingxi Chen, Ke Hao, Yabo Chen, Zhengxue Cheng, Rong Xie, Li Song, Haibin Huang, Chi Zhang, and Xuelong Li. 2026a. Full-4D: Generating Full-Scope 4D Scenes from a Single-View Video. arXiv:2605.25500 [cs.CV] https://arxiv.org/abs/2605.25500

Xiangyu Chen, Jixiang Luo, Jingyu Xu, Fangqiu Yi, Chi Zhang, and Xuelong Li. 2026b. Generative video compression: towards 0.01% compression rate for video transmis sion. Vicinagearth 3, 1 (2026), 7.

Yabo Chen, Jiemin Fang, Yuyang Huang, Taoran Yi, Xiaopeng Zhang, Lingxi Xie, Xinggang Wang, Wenrui Dai, Hongkai Xiong, and Qi Tian. 2024a. Cascade-Zero123: One Image to Highly Consistent 3D with Self-Prompted Nearby Views. arXiv:2312.04424 [cs.CV] https://arxiv.org/abs/2312.04424

Yabo Chen, Yuanzhi Liang, Jiepeng Wang, Tingxi Chen, Junfei Cheng, Zixiao Gu, Yuyang Huang, Zicheng Jiang, Wei Li, Tian Li, et al. 2025a. TeleWorld: Towards Dynamic Multimodal Synthesis with a 4D World Model. arXiv preprint arXiv:2601.00051 (2025).

Yabo Chen, Chen Yang, Jiemin Fang, Xiaopeng Zhang, Lingxi Xie, Wei Shen, Wenrui Dai, Hongkai Xiong, and Qi Tian. 2024b. LiftImage3D: Lifting Any Single Image to 3D Gaussians with Video Generation Priors. arXiv:2412.09597 [cs.CV] https: //arxiv.org/abs/2412.09597

Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. 2026. LoL: Longer than Longer, Scaling Video Generation to Hour. arXiv preprint arXiv:2601.16914 (2026).

Dan Friedman and Adji Bousso Dieng. 2023. The Vendi Score: A Diversity Evaluation Metric for Machine Learning. TMLR (2023).

Michael Fuest, Pingchuan Ma, Ming Gui, Johannes Schusterbauer, Vincent Tao Hu, and Björn Ommer. 2026. Difusion models and representation learning: A survey. TPAMI (2026).

Rohit Gandikota and David Bau. 2026. Distilling diversity and control in difusion models. In WACV. 1304–1313.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. 2026. Mean flows for one-step generative modeling. NeurIPS 38 (2026), 75460–75482.

Zixiao Gu, Yabo Chen, Xunzhi Xiang, Yu He, Haibin Huang, Chi Zhang, Yunbo Wang, and Xuelong Li. 2026. Search-to-World: Evaluation of 3D World Delivery from User Request through Web Search. arXiv:2609.07605 [cs.CV] https://arxiv.org/abs/2609. 07605

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. 2023. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers. In ICLR.

Li Hu. 2024. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. In CVPR. 8153–8163.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. 2025b. Self forcing: Bridging the train-test gap in autoregressive video difusion. arXiv preprint arXiv:2506.08009 (2025).

Yuyang Huang, Yabo Chen, Wenrui Dai, Ziyang Zheng, Haibin Huang, Chi Zhang, Junni Zou, Hongkai Xiong, and Xuelong Li. 2026. CineWeaver: Training-Free Reference-Controllable Multi-Shot Long Video Generation for Cinematic Story telling. arXiv:2607.26529 [cs.CV] https://arxiv.org/abs/2607.26529

Yuyang Huang, Yabo Chen, Li Ding, Xiaopeng Zhang, Wenrui Dai, Junni Zou, Hongkai Xiong, and Qi Tian. 2025a. Im-zero: Instance-level motion controllable video generation in a zero-shot manner. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 7265–7275.

Yuyang Huang, Yabo Chen, Yuchen Liu, Xiaopeng Zhang, Wenrui Dai, Hongkai Xiong, and Qi Tian. 2024a. DomainFusion: Generalizing to unseen domains with latent difusion models. In European conference on computer vision. Springer, 480–498.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. 2024b. Vbench: Comprehensive benchmark suite for video generative models. In CVPR. 21807– 21818.

Francisco Ibarrola and Kazjon Grace. 2024. Measuring diversity in co-creative image generation. arXiv preprint arXiv:2403.13826 (2024).

Weinan Jia, Yuning Lu, Mengqi Huang, Hualiang Wang, Binyuan Huang, Nan Chen, Mu Liu, Jidong Jiang, and Zhendong Mao. 2026. MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation. In ICLR.

Tahira Kazimi, Connor Dunlop, and Pinar Yanardag. 2025. Diverse Video Generation with Determinantal Point Process-Guided Policy Optimization. arXiv preprint arXiv:2511.20647 (2025).

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework

for large video generative models. arXiv preprint arXiv:2412.03603 (2024).

Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. 2026. Autoregressive adversarial post-training for real-time interactive video generation. NeurIPS 38 (2026), 41061–41086.

Dongyang Liu, Peng Gao, David Liu, Ruoyi Du, Zhen Li, Qilong Wu, Xin Jin, Sihan Cao, Shifeng Zhang, Steven HOI, and Hongsheng Li. 2026a. Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield. In ICLR.

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. 2026b. Rolling Forcing: Autoregressive Long Video Difusion in Real Time. In ICLR.

Dakuan Lu, Jiaqi Zhang, Cheng Yuan, Jiawei Shao, and Xuelong Li. 2025b. The Law of Multi-Model Collaboration: Scaling Limits of Model Ensembling for Large Language Models. arXiv preprint arXiv:2512.23340 (2025).

Yanzuo Lu, Yuxi Ren, Xin Xia, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Andy J Ma, Xiaohua Xie, and Jian-Huang Lai. 2025a. Adversarial distribution matching for difusion distillation towards eficient image and video synthesis. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 16818–16829.

Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. 2023. Dif-instruct: A universal approach for transferring knowledge from pre-trained difusion models. NeurIPS 36 (2023), 76525–76546.

Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. 2024. One-step difusion distillation through score implicit matching. NeurIPS 37 (2024), 115377–115408.

Thuan Hoang Nguyen and Anh Tran. 2024. Swiftbrush: One-step text-to-image difusion model with variational score distillation. In CVPR. 7807–7816.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2024. DINOv2: Learning Robust Visual Features without Supervision. TMLR (2024). Featured Certification.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In ICML. PmLR, 8748–8763.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. 2024. Adversaria difusion distillation. In ECCV. Springer, 87–103.

Jiawei Shao and Xuelong Li. 2025. Ai flow at the network edge. IEEE Network 40, 1 (2025), 330–336.

Hui Shen, Jingxuan Zhang, Boning Xiong, Rui Hu, Shoufa Chen, Zhongwei Wan, Xin Wang, Yu Zhang, Zixuan Gong, Guangyin Bao, et al. 2025. Eficient difusion models: A survey. arXiv preprint arXiv:2502.06805 (2025).

Quanyue Song, Zhizhi Guo, Yishan He, Zhihao Wang, Zhixiang He, Chi Zhang, Caigu Jiang, and Xuelong Li. 2026. SyncDIT: audio-visual aligned video generation with audio synchronization feature. Vicinagearth 3, 1 (2026), 5.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency Models. arXiv:2303.01469 [cs.LG] https://arxiv.org/abs/2303.01469

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. 2025. WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Model. arXiv preprint (2025).

Christian Szegedy, Vincent Vanhoucke, Sergey Iofe, Jon Shlens, and Zbigniew Wojna. 2016. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2818–2826.

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. 2025. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211 (2025).

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314 (2025).

Haoyuan Wang, Yabo Chen, Haibin Huang, Chi Zhang, and Xuelong Li. 2026. Directing the World: Fast Autoregressive Video Generation with Compositional Human-Camera Control. arXiv:2606.27964 [cs.CV] https://arxiv.org/abs/2606.27964

Wenhao Wang and Yi Yang. 2024. Vidprom: A million-scale real prompt-gallery dataset for text-to-video difusion models. NeurIPS 37 (2024), 65618–65642.

Tao Wen, Jiepeng Wang, Yabo Chen, Shugong Xu, Chi Zhang, and Xuelong Li. 2025. Metric-Solver: Sliding Anchored Metric Depth Estimation from a Single Image. arXiv:2504.12103 [cs.CV] https://arxiv.org/abs/2504.12103

Tianhe Wu, Ruibin Li, Lei Zhang, and Kede Ma. 2026. Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis. arXiv preprint arXiv:2602.03139 (2026).

Xunzhi Xiang, Yabo Chen, Guiyu Zhang, Zhongyu Wang, Zhe Gao, Quanming Xiang, Gonghu Shang, Junqi Liu, Haibin Huang, Yang Gao, et al. 2025. Macro-from-micro planning for high-quality and parallelized autoregressive long video generation. arXiv preprint arXiv:2508.03334 (2025).

Xunzhi Xiang, Zixuan Duan, Yabo Chen, Zhengxuan Wei, Guiyu Zhang, Zixiao Gu, Zhe Gao, Haibin Huang, Chi Zhang, Qi Fan, and Xuelong Li. 2026a. VideoWeave: Unlocking Geometric Consistency in Video Generation via Joint Geometry-Video Modeling. arXiv:2606.14162 [cs.CV] https://arxiv.org/abs/2606.14162

Xunzhi Xiang, Zixuan Duan, Guiyu Zhang, Haiyu Zhang, Zhe Gao, Junta Wu, Shaofeng Zhang, Tengfei Wang, Qi Fan, and Chunchao Guo. 2026b. Pathwise Test-Time Cor rection for Autoregressive Long Video Generation. arXiv preprint arXiv:2602.05871 (2026).

Xunzhi Xiang and Qi Fan. 2025. Make It Eficient: Dynamic Sparse Attention for Autoregressive Image Generation. arXiv:2506.18226 [cs.CV] https://arxiv.org/abs 2506.18226

Yilun Xu, Weili Nie, and Arash Vahdat. 2025. One-step difusion models with �- divergence distribution matching. arXiv preprint arXiv:2502.15681 (2025).

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Ying-Cong Chen, Yao Lu, Song Han, and Yukang Chen. 2026. LongLive: Real-time Interactive Long Video Generation. In ICLR.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. 2024a. Improved distribution matching distillation for fast image synthesis. NeurIPS 37 (2024), 47455–47487.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. 2024b. One-step difusion with distribution matching distillation. In CVPR. 6613–6623.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, El Shechtman, and Xun Huang. 2025. From slow bidirectional to fast autoregressive video difusion models. In CVPR. 22963–22974.

Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. 2025. Context as memory: Scene-consistent interactive long video generation with memory retrieval. In Proceedings ofthe SIGGRAPH Asia 2025 Conference Papers. 1–11.

Cheng Yuan, Zhenyu Jia, Jiawei Shao, and Xuelong Li. 2026a. Enhancing Neural Video Compression of Static Scenes with Positive-Incentive Noise. arXiv preprint arXiv:2603.06095 (2026).

Shenghai Yuan, Yuanyang Yin, Zongjian Li, Xinwei Huang, Xiao Yang, and Li Yuan. 2026b. Helios: Real Real-Time Long Video Generation Model. arXiv preprint arXiv:2603.04379 (2026).

Xianghao Zang, Zijian Jiang, Jiarong Cheng, Qianrui Teng, Ying He, Yuxuan Mu, Chao Ban, Huayu Zhang, Lanxiang Zhou, Zerun Feng, et al. 2026. Instruction-based image editing: a survey on data, models, evaluation, and applications. Vicinagearth 3, 1 (2026), 3.

Guiyu Zhang, Yabo Chen, Xunzhi Xiang, Junchao Huang, Zhongyu Wang, and Li Jiang. 2026c. SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 11127–11137.

Xin Zhang, Yabo Chen, Zixuan Duan, Haibin Huang, Chi Zhang, Feng Xu, and Xuelong Li. 2026a. TourPhysics: Bringing Physics to World Models for Exploration and Manipulation from a Single Image. arXiv:2609.04911 [cs.CV] https://arxiv.org/abs/ 2609.04911

Xin Zhang, Yabo Chen, Yijie Fang, Wanying Qu, Haibin Huang, Chi Zhang, Feng Xu, and Xuelong Li. 2026b. PhysOmni: Physics-Grounded Multi-Object Scene Generation from a Single Image with Real-Time Interaction. arXiv:2605.20290 [cs.GR] https: //arxiv.org/abs/2605.20290

Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, and Hai Huang. 2025. Adversarial Score identity Distillation: Rapidly Surpassing the Teacher in One Step. In ICLR.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. 2026. Causal Forcing: Autoregressive Difusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation. arXiv preprint arXiv:2602.02214 (2026).