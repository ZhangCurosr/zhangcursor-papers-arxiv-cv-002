# Optimize Your Sampling: Tuned Difusion Sampling with Bayesian Optimization

Travis Zhang Cornell University

Christian Belardi<sup>∗</sup> Cornell University

Justin Lovelace Cornell University

Jin Peng Zhou Cornell University

Saebyeol Shin Cornell University

Carla P. Gomes Cornell University

Kilian Q. Weinberger

Cornell University

tz98@cornell.edu

ckb73@cornell.edu

## Abstract

Sampling from a difusion model typically requires many forward passes through a large neural network, making generation computationally expensive. While much work has focused on eficient solvers and samplers, comparatively little attention has been paid to selecting the sampling timesteps themselves. A recent line of work optimizes theoretically derived surrogates for sample quality rather than the quality metric itself. We propose Optimizing Your Sampling (OYS), which instead treats timestep selection as a black-box optimization problem, optimizing the target metric directly with Bayesian optimization. OYS outperforms both the default schedules and those of Align Your Steps on text-to-image generation, and improves over the default schedules on inpainting and other image tasks, in both quantitative and human evaluations. OYS requires no additional training, is applicable even to distilled models, and improves both simple and sophisticated samplers such as Euler and DPM-Solver++. A 5-step OYS schedule retains 89%–94% of the quality of a 50-step schedule while reducing inference cost by 10x.

## 1 Introduction

Difusion models (Ho et al., 2020; Song et al., 2021; Sohl-Dickstein et al., 2015) have achieved state-of-the-art performance across a wide range of generative tasks, including text-to-image synthesis (Ramesh et al., 2022; Rombach et al., 2022; Saharia et al., 2022b; Peebles & Xie, 2023), super resolution (Saharia et al., 2023; 2022a), inpainting (Lugmayr et al., 2022; Corneanu et al., 2024), and video generation (Ho et al., 2022; Blattmann et al., 2023). However, the iterative denoising process requires many forward passes through large neural networks, making generation computationally expensive.

A happy daffodil with big eyes, multiple leaf arms and vine legs, rendered in 3D Pixar style.

A painting of a starship landing by a temple, created by Hubert Robert.

Prompt

A white Persian cat   
wearing a peacock   
feather headdress   
and surrounded by flowers, in a magical realism painting.

Two cats, one grey and one black, are wearing steampunk attire and standing in front of a ship in a heavily detailed painting.

Rosario Dawson minimalist   
portrait by Jean   
Giraud in a comic book style.

![](images/fcc3df58019a2b26102e793e8345fcac9ffdf2860ad084c541d601b07e529b43.jpg)

![](images/918b475ed129fe6736141d4c518f25c02a9dff8f1af486c734f99230e981ad95.jpg)  
DeepFloyd

Default Y<sup>S</sup> <sub>O</sub><sup>YS</sup>

A lady in a purple dress sitting in a tree - concept art.

![](images/e05536c5d9b6ab8df2ac86f3737db7767ee54aa75ff2d6207b7dbcf339ac7c7d.jpg)

SDXL  
![](images/c189e2b8d1eaef20e1bda4e0cd8c5f81d35be5f290fcb0aa7e6f2f1bb145e0bd.jpg)

![](images/401c1360bcf32f774979af257f6c48528c00de2904d39f1a15320d3ead9a7d75.jpg)

![](images/d208b9157b70fcebdc9f8d7c62d480ea824d1d6534ead485546a3122324ce399.jpg)  
SDv1.5

Figure 1: Qualitative comparisons between Default, AYS (Sabour et al., 2024), and ours (OYS) using a 5 step sampling schedule. All prompts are held-out test examples, unseen during schedule optimization for either method.

Much of the efort to reduce this cost has focused on designing eficient samplers (Liu et al., 2022; Lu et al., 2022a;b), and developing few step models such as distilled models (Salimans & Ho, 2022; Meng et al., 2023; Zhou et al., 2024; Xie et al., 2024; Starodubcev et al., 2025; Gu et al., 2023), consistency models (Song et al., 2023; Luo et al., 2023; Lu & Song, 2024), and meanflow models (Geng et al., 2026a;b). Far less attention has been given to the choice of sampling timesteps, that is, which timesteps in the reverse difusion process to use. Yet this choice has a substantial impact on quality, particularly at low step budgets where the default schedule degrades rapidly.

Prior work, Align Your Steps (AYS) (Sabour et al., 2024), addresses this by formulating timestep selection as minimization of a KL divergence upper bound (KLUB) between the true generative process and its discretization. Fully minimizing this bound, however, need not minimize the mismatch between output distributions. Therefore AYS requires early stopping from a heuristic initialization (Sabour et al., 2024). All the released AYS schedules, which were optimized at 10 steps, remain close to the default schedule in log-SNR space. We find that this is costly at low step budgets, where globally reallocating steps across the full timestep range yields larger gains than local refinement allows.

We propose Optimizing Your Sampling (OYS), which instead treats timestep selection as a black-box optimization problem and solves it with Bayesian optimization (Kushner, 1964; Žilinskas, 1975; Močkus, 1975; Jones et al., 1998; Osborne et al., 2009; Garnett, 2023). OYS searches the full configuration space without requiring gradient information, allowing the optimizer to reallocate steps across the entire timestep range rather than refining locally. It requires no additional training and applies to any difusion model.

OYS consistently outperforms both AYS and the log-linearly downsampled default schedule on text-to-image generation, and the default schedule on inpainting and inverse image tasks, verified by both quantitative metrics and human evaluation; Figure 1 shows qualitative comparisons. Inspecting the optimized schedules reveals a consistent pattern: OYS allocates more steps to high-noise timesteps, in contrast to the approximately uniform log-SNR spacing of both AYS and the default schedule. OYS retains much of the quality of a 50-step schedule using only 5 steps. Further, OYS is eficient: for SDXL, tuning plateaus at approximately 17K five-step generations, 85K forward passes — over an order of magnitude fewer than the upper bound Sabour et al. (2024) report for AYS’s tuning cost.

![](images/f556a4c9d0bd4a138b6a25f1ec552cb7202ee443608bafec37ac5e7228563ff1.jpg)  
Figure 2: Overview of our Optimize Your Sampling (OYS) framework using Bayesian Optimization. The process begins with Acquisition Maximization (top left), where promising sampling configurations are identified. The selected configuration undergoes Black Box Metric Evaluation (right). This consists of generating images with the difusion model and scoring them using metrics like FID and HPS. Results update the surrogate model’s posterior distribution (bottom left), refining our understanding of the configuration space. This cycle repeats until convergence or budget exhaustion, eficiently discovering efective sampling parameters.

## 2 Related Work

Several prior works have explored customized sampling schedules or configurations. Karras et al. (2022) present a unifying framework for difusion model training and sampling that defines a number of parameters that control various aspects of both stochastic and deterministic sampling, including the schedule. The parameters $\sigma _ { \mathrm { m i n } }$ and $\sigma _ { \mathrm { m a x } }$ specify the bounds of the noise range, while $\rho$ controls how step sizes are distributed. However, finding the optimal configuration requires an expensive grid search, with each evaluation requiring 50,000 image generations. Karras et al. (2024) extend this framework with additional parameters including exponential moving average (EMA) length and guidance strength, again relying on grid search.

Lugmayr et al. (2022) were among the first to propose a custom sampling schedule, designing a denoising schedule for inpainting that repeatedly steps backward and forward in time — tracing a zigzag through the difusion process — improving coherence between masked and unmasked regions at the cost of additional function evaluations.

Watson et al. (2021) instead treat the schedule itself as an object of optimization, using dynamic programming to select the K-step subsequence that maximizes an evidence lower bound; the resulting schedules improve likelihood but not perceptual quality. Watson et al. (2022) address this mismatch with Diferentiable Difusion Sampler Search (DDSS), which optimizes the degrees of freedom of a parametric family of samplers by backpropagating through the sampling process to maximize a perceptual score. This imposes two costs. First, the objective must admit gradients, so they optimize the Kernel Inception Distance as a diferentiable stand-in for sample quality. Second, unrolling the sampling chain requires retaining its forward-pass state, whose memory grows linearly in the number of sampling steps and, as the authors note, quickly becomes infeasible for large denoiser architectures; DDSS must rematerialize the score function calls, recomputing them during the backward pass to trade this memory against additional compute. OYS shares the goal of optimizing perceptual quality directly, but treats the objective as a black box: each evaluation is an ordinary forward sampling pass, with no gradients flowing through the sampler. It therefore optimizes the target metric itself rather than a diferentiable proxy, avoids both the memory of unrolling the sampler and the compute of rematerializing it, and accommodates non-diferentiable operations such as the rounding of timesteps to integers for discrete-time models.

Sabour et al. (2024) propose Align Your Steps (AYS), which formulates schedule optimization as minimizing the Kullback-Leibler divergence Upper Bound (KLUB) between the true generative SDE and its discretization. The KLUB decomposes as a sum over the intervals between consecutive schedule points. AYS minimizes this total via a zeroth-order procedure that iteratively adjusts each intermediate timestep locally, with endpoints fixed and the schedule initialized from a heuristic schedule. They demonstrate that hand-crafted schedules such as Karras et al. (2022) are suboptimal, and their tuned schedules improve sample quality. However, because the KLUB is an upper bound rather than the output mismatch itself, minimizing it need not minimize that mismatch (Sabour et al., 2024), and their procedure relies on early stopping to avoid over-optimizing it. Empirically, the resulting schedules remain close to the default schedule in log-SNR space (see Figure 5). The released AYS schedules are 10-step; we adapt them to fewer steps via their prescribed log-linear interpolation.

Two further methods revisit timestep selection. Concurrent with our work, Huang et al. (2026) propose Adaptive Reparameterized Time (ART), an optimal control formulation that treats the speed of the sampler’s clock as a control variable, redistributing computation along the trajectory so as to minimize the aggregate Euler discretization error, and solve the resulting problem with continuous-time reinforcement learning. In recent work, Zhu et al. (2026) propose the Hierarchical Schedule Optimizer (HSO), a bi-level method that alternates a global search over schedule initializations with local refinement of a midpoint error proxy, penalizing pathologically close timesteps to remain robust at very low step counts.

Both share the defining feature of AYS: they optimize a theoretically derived surrogate for sample quality — the KLUB for AYS, Euler discretization error for ART, and the midpoint error proxy for HSO — rather than the quantity of interest itself. Doing so makes their search inexpensive, as no images need be generated while optimizing, but it binds the resulting schedule to how faithfully the surrogate tracks perceptual quality. OYS sits at the opposite end of this trade-of. Because it evaluates the target metric directly, it incurs the cost of generating images during tuning, but it optimizes exactly the criterion the schedule is ultimately judged on, and applies unchanged to objectives for which no tractable surrogate exists — human preference scores, LPIPS, or task-specific reconstruction error. This cost is paid once per model and task and amortized over all subsequent sampling. Our approach also extends naturally beyond the timesteps themselves to the surrounding sampling parameters, such as guidance strength and EMA length, which surrogate-based objectives do not model.

## 3 Optimize Your Sampling (OYS)

In this section, we describe the OYS framework. We present an overview of the optimization loop in Figure 2.

At a high level, for a given task—such as text-to-image generation or inpainting—we assume access to a set of inputs X for tuning, a fixed pretrained difusion model m, and a black-box evaluation metric e that quantifies performance. OYS iteratively optimizes a set of sampling parameters $\mathbf { p } = \{ p _ { 1 } , \dots , p _ { M } \}$ using Bayesian optimization, continuing until the optimization either converges or the budget is expended.

Optimizing Diferent Difusion Formulations. Difusion models can be broadly categorized into two formulations: discrete-time, which defines the difusion process as a fixed-length Markov chain of noise-addition steps, and continuous-time, which defines it as a diferential equation over a continuous noise schedule. In both formulations, time parametrizes the noise level, and sampling requires selecting an ordered sequence of timesteps, the sampling schedule, at which to feed through the model. How this sampling schedule is represented varies: most models leave it unparameterized, specifying the timesteps directly, while some continuous-time models define it as a parametric family of functions with a few tunable parameters. We address each case below.

Direct schedule optimization. Most difusion models do not prescribe a parametric family for the sampling schedule. For these models, we optimize the schedule timesteps directly.

The key challenge is parameterizing the schedule for continuous optimization. While there is no strict requirement that sampling schedules be monotonically decreasing, standard practice suggests that the denoising trajectory should proceed from high to low noise. We enforce this with a cumulative product reparameterization: for each step k in a K-step schedule, we optimize a continuous parameter $p _ { k } \in [ 0 , 1 ]$ and define the schedule as

$$
\begin{array} { l } { { \displaystyle { \boldsymbol { s } } ( { \bf p } ) = \left[ t _ { \operatorname* { m a x } } \prod _ { i = 1 } ^ { k } p _ { i } \right] _ { k = 1 } ^ { K } } } \\ { { \displaystyle ~ = [ t _ { \operatorname* { m a x } } p _ { 1 } , ( t _ { \operatorname* { m a x } } p _ { 1 } p _ { 2 } ) , \ldots , ( t _ { \operatorname* { m a x } } p _ { 1 } p _ { 2 } \ldots { } . . p _ { K } ) ] } } \end{array}
$$

where $t _ { \mathrm { m a x } }$ is the maximum timestep and $K \leq M$ to allow for additional hyperparameters beyond the schedule. Since each $p _ { k }$ multiplies the running product by a value in $[ 0 , 1 ]$ , the schedule is monotonically decreasing by construction. Each parameter $p _ { k }$ for $k > 1$ is the ratio between consecutive timesteps, $t _ { k } / t _ { k - 1 }$

For discrete-time models, whose timesteps must belong to a fixed set of integers (typically $\{ 0 , \ldots , 9 9 9 \} )$ , we apply rounding, denoted by $q ,$ after scaling:

$$
\begin{array} { l } { \displaystyle { s ( { \bf p } ) = \left[ q \Big ( t _ { \mathrm { m a x } } \prod _ { i = 1 } ^ { k } p _ { i } \Big ) \right] _ { k = 1 } ^ { K } } } \\ { \displaystyle ~ = \left[ q \big ( t _ { \mathrm { m a x } } p _ { 1 } \big ) , q \big ( t _ { \mathrm { m a x } } p _ { 1 } p _ { 2 } \big ) , \dots , q \big ( t _ { \mathrm { m a x } } p _ { 1 } p _ { 2 } \dots p _ { K } \big ) \right] . } \end{array}
$$

We note that this parameterization does not induce a uniform distribution over valid monotonic schedules. Because later timesteps are products of more [0, 1] terms, they concentrate near smaller values, biasing the search toward schedules that spend more steps at low noise levels. In practice, the Bayesian optimization surrogate learns to compensate for this non-uniformity, and we find that OYS consistently discovers schedules that allocate more steps to high-noise (large timestep) regions despite this bias, see Figure 5.

Parametric schedules. The EDM family of difusion models (Karras et al., 2022; 2024) prescribes a functional form for their sampling schedules, parameterized by $\sigma _ { \mathrm { m i n } } , \ \sigma _ { \mathrm { m a x } } ,$ and $\rho .$ Karras et al. (2024) additionally tune the EMA length and classifier-free guidance strength. Since all parameters are real-valued and constrained to closed intervals, we treat each as an element in p and optimize them jointly. In this setting the timesteps themselves are tuned indirectly, through the parameters of the schedule’s functional form.

Bayesian Optimization. Our goal is to find the sampling configuration, $\mathbf { p } ,$ that minimizes the evaluation metric, e, on our tuning set, X , for our model, m,

$$
\mathbf { p } ^ { * } = \underset { \mathbf { p } \in \mathcal { P } } { \arg \operatorname* { m i n } } e ( m ( \mathcal { X } ; \mathbf { p } ) ) .
$$

For metrics where higher is better, such as HPS, we minimize the negated metric; we therefore phrase optimization as minimization throughout.

We use Bayesian optimization (Kushner, 1964; Žilinskas, 1975; Močkus, 1975; Jones et al., 1998; Osborne et al., 2009; Garnett, 2023), which sequentially selects and evaluates configurations using a Gaussian process to approximate the function $f ( \mathbf { p } ) = e ( m ( \mathcal { X } ; \mathbf { p } ) )$ . The Gaussian process, fit to observed evaluations $\mathbf { \bar { \mathcal { D } } } = ( \mathbf { p } ^ { ( j ) } , y ^ { ( j ) } ) _ { j = 1 } ^ { N }$ , provides a posterior distribution over the objective function, where $y ^ { ( j ) }$ is the observed value of sampling configuration $\mathbf { p } ^ { ( j ) }$ . This posterior afords both mean and uncertainty estimates which are used to define an acquisition function that guides the search for promising sampling configurations. The initial configurations are chosen via Sobol sampling, with the number of Sobol samples set to twice the number of parameters. After evaluating the Sobol-sampled configurations, we fit a Gaussian process and switch to the qLogNEI (Ament et al., 2023; Letham et al., 2019) acquisition function to select new candidate configurations.

## 4 Experiments

We benchmark sampling configurations found with OYS across a variety of image-domain tasks, comparing with relevant baselines when applicable. In all cases when we downsample a schedule, we use the log-linear interpolation procedure that Sabour et al. (2024) prescribe for adapting their schedules to other step counts. As the released AYS schedules are all 10-step schedules (Sabour et al., 2024), we obtain the few-step AYS baselines via their downsampling procedure. For all experiments, we evaluate on a held-out set of examples.

Evaluation Metrics. We evaluate with standard image quality metrics such as Fréchet Inception Distance (FID) (Heusel et al., 2017) and Peak Signal-to-Noise Ratio (PSNR) when applicable. However, these metrics do not capture alignment between text prompts and generated images.

To address this, Wu et al. (2023) introduced the Human Preference Dataset v2 (HPD v2) and fine-tuned the ViT-L/14 version of a CLIP model on the dataset. Using this fine-tuned model, they define the Human Preference Score for a text-image pair (T, I) as,

$$
\mathrm { H P S } ( T , I ) = \omega \frac { \mathcal { E } _ { T } ( T ) } { | | \mathcal { E } _ { T } ( T ) | | } \frac { \mathcal { E } _ { I } ( I ) } { | | \mathcal { E } _ { I } ( I ) | | }
$$

where ${ \mathcal { E } } _ { T }$ and ${ \mathcal { E } } _ { I }$ represent the CLIP text and image encoder respectively, and ω is a learned scaling parameter. Given a prompt T and images $I _ { 1 }$ and $I _ { 2 }$ generated by two competing methods, the predicted preference win rate is computed from their HPS scores using the Bradley–Terry model,

$$
\operatorname { \cal P } ( I _ { 1 } \succ I _ { 2 } \mid T ) = \operatorname { S o f t m a x } \bigl ( \operatorname { H P S } ( T , I _ { 1 } ) , \operatorname { H P S } ( T , I _ { 2 } ) \bigr ) _ { I _ { 1 } } .
$$

We report empirical win rates from per-prompt comparisons: for each held-out prompt, we generate one image with each of the two schedules being compared, score both with HPS, and award the win to the higher-scoring image; the win rate is the fraction of prompts won. Unless otherwise noted, the comparison schedule is the default schedule at the same step count.

Text-to-Image. We evaluate OYS on six text-to-image models. On COCO Captions (Chen et al., 2015), we tune on the training split and evaluate on the validation split for DeepFloyd (at StabilityAI, 2023), Stable Difusion XL (Podell et al., 2024), and Stable Difusion v1.5 (Rombach et al., 2022) — the models with published AYS schedules (Sabour et al., 2024), our primary point of comparison — as well as SDXL-Turbo (Sauer et al., 2025), a model distilled for few-step sampling that tests whether schedule tuning complements distillation. On DifusionDB (Wang et al., 2022), we tune FLUX.1-dev (Labs et al., 2025) and QwenImage (Wu et al., 2025) on one subset and evaluate on a held-out subset, comparing against their default schedules. We use Bayesian optimization to optimize the sampling timesteps to maximize HPS (Wu et al., 2023). For each round, we calculate HPS with a batch size of 128. Following Sabour et al. (2024), we fix the sampler for DeepFloyd, Stable Difusion XL, and Stable Difusion v1.5 to DPM-Solver++ (Lu et al., 2022b), while FLUX.1-dev, QwenImage, and SDXL-Turbo use the Euler Discrete sampler.

We report results on COCO Captions in Table 1 and on DifusionDB in Table 2. As shown in Table 1, at 10 steps schedule choice matters little: log-linearly downsampling the default schedule remains competitive with AYS across all three models. At 5 steps, however, performance degrades substantially and the schedules meaningfully separate. We therefore focus on this aggressive low-step regime, where schedule choice has the greatest leverage on quality. OYS discovers improved sampling schedules across all models, achieving higher win rates against the default and higher average HPS scores.

For SDXL-Turbo, the 3-step tuned schedule improves over the 3-step default schedule. For FLUX.1-dev and QwenImage, shown in Table 2, tuned 5-step schedules outperform default schedules, requiring only 5,376 and

<table><tr><td>Model</td><td>Method</td><td>Steps</td><td>HPS</td><td>HPS Average↑ Win Rate↑</td></tr><tr><td rowspan="6">DeepFloyd</td><td>Default</td><td>50</td><td>0.267</td><td>-</td></tr><tr><td>Default</td><td>10</td><td>0.258</td><td>-</td></tr><tr><td>AYS</td><td>10</td><td>0.261</td><td>-</td></tr><tr><td>Default</td><td>5</td><td>0.215</td><td></td></tr><tr><td>AYS</td><td>5</td><td>0.225</td><td>65.39%</td></tr><tr><td>OYS</td><td>5</td><td>0.252</td><td>83.99%</td></tr><tr><td rowspan="6">Stable Diffusion v1.5</td><td>Default</td><td>50</td><td>0.262</td><td>-</td></tr><tr><td>Default</td><td>10</td><td>0.258</td><td>-</td></tr><tr><td>AYS</td><td>10</td><td>0.258</td><td>一</td></tr><tr><td>Default</td><td>5</td><td>0.230</td><td></td></tr><tr><td>AYS</td><td>5</td><td>0.231</td><td>52.30%</td></tr><tr><td>OYS</td><td>5</td><td>0.238</td><td>59.29%</td></tr><tr><td rowspan="6">Stable Diffusion XL</td><td>Default</td><td>50</td><td>0.275</td><td>-</td></tr><tr><td>Default</td><td>10</td><td>0.266</td><td>-</td></tr><tr><td>AYS</td><td>10</td><td>0.264</td><td>-</td></tr><tr><td>Default</td><td>5</td><td>0.229</td><td></td></tr><tr><td>AYS</td><td>5</td><td>0.210</td><td>25.29%</td></tr><tr><td>OYS</td><td>5</td><td>0.245</td><td>70.70%</td></tr><tr><td rowspan="2">SDXL Turbo</td><td>Default</td><td>3</td><td>0.287</td><td>一</td></tr><tr><td>OYS</td><td>3</td><td>0.298</td><td>83.35%</td></tr></table>

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">Steps</td><td>HPS</td><td>HPS</td></tr><tr><td></td><td>Average↑ Win Rate↑</td></tr><tr><td rowspan="2">FLUX.1 dev</td><td>Default</td><td>5</td><td>0.251</td><td></td></tr><tr><td>OYS</td><td>5</td><td>0.283</td><td>82.69%</td></tr><tr><td rowspan="2">Qwen Image</td><td>Default</td><td>5</td><td>0.174</td><td></td></tr><tr><td>OYS</td><td>5</td><td>0.193</td><td>77.19%</td></tr></table>

Table 2: Text-to-Image on DifusionDB. Comparison of the default and OYS schedules on HPS Average and HPS Win Rate for FLUX.1-dev and QwenImage, tuned and evaluated on disjoint subsets of DifusionDB. Both models use the Euler Discrete sampler. Win Rates are computed with respect to the default schedule at the same step count. Because these models are evaluated on a diferent prompt set, their HPS Averages are not directly comparable to those in Table 1, and because both are run far below their default step counts, diferences between the two models do not reflect their relative quality at full budgets.  
Table 1: Text-to-Image on COCO Captions. Comparison of diferent models and methods on HPS Average and HPS Win Rate, evaluated on the COCO Captions validation split. Win Rates are computed with respect to the baseline scheduler at the same step count. DeepFloyd, Stable Difusion v1.5, and Stable Difusion XL use DPM-Solver++. For SDXL-Turbo, we benchmark the Euler Discrete (ED) sampler; Euler Ancestral (EA) results are included in the Appendix for completeness.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">PSNR↑</td><td>HPS</td><td>HPS</td></tr><tr><td></td><td>Average↑ Win Rate↑</td></tr><tr><td rowspan="2">SDXL</td><td>Default</td><td>66.59</td><td>0.203</td><td></td></tr><tr><td>OYS</td><td>73.47</td><td>0.240</td><td>93.40%</td></tr><tr><td rowspan="2">SDv1.5</td><td>Default</td><td>64.46</td><td>0.213</td><td></td></tr><tr><td>OYS</td><td>67.34</td><td>0.237</td><td>85.10%</td></tr></table>

Table 3: Inpainting. PSNR, Average HPS, and HPS Win Rate on SDXL and SDv1.5 comparing Default and OYS schedules. HPS Win Rates are with respect to the Default schedule.

2,560 generations respectively—equivalent in cost to just 960 and 256 generations at the models’ respective 28- and 50-step default budgets. The two models’ absolute HPS should therefore not be read as a model comparison: all values in Table 2 reflect a 5-step budget, far below those defaults. These results span standard and distilled models, discrete-time and continuous-time formulations, and multiple samplers, suggesting that the benefits of schedule tuning with OYS are broadly applicable.

To assess whether OYS’s improvements correspond to human-perceived quality rather than being specific to the tuning metric, we conducted a human evaluation comparing generations from the 5-step schedules for AYS, OYS, and Default using SDXL. We crowdsourced responses from 58 participants assessing both image fidelity and alignment. Each participant received randomly shufled pairs of images generated from either AYS, OYS, or Default, and was asked to choose whether image 1, image 2, or tie better answers the question. We provide an example of the rating interface in the Appendix. As shown in Figure 3, OYS significantly outperforms both AYS and Default in image quality and alignment. Significance is assessed with a two-sided exact binomial test on the win and lose counts of each pairing, with ties excluded; every comparison is significant at $p < 0 . 0 0 1$ . For quality, OYS achieves a 68.8% win rate against AYS and 70.6% against Default, with ties split equally between methods. For alignment, OYS achieves a 62.9% win rate against AYS and 66.8% against Default. When AYS is log-linearly downsampled to 5 timesteps, it loses to the 5-step default schedule for SDXL. These human preference results are corroborated by performance on metrics that were never the tuning objective. Tuning on HPS improves FID substantially on DeepFloyd and SDXL. On SDv1.5, all 5-step schedules achieve comparable FID; we do not read much into diferences this small, as zero-shot COCO FID only loosely tracks perceived quality (Hoogeboom et al., 2025). In no case does optimizing for human preference degrade distributional quality (see Table 6 in the Appendix). As shown in the following sections, tuning on LPIPS for inpainting improves both PSNR and HPS (see Table 3), and tuning on MSE for the inverse image tasks improves HPS (see Table 4). Together, these results indicate that OYS’s gains reflect genuine improvements in image quality.

![](images/c2294d048acfb671b3ef905a405aa5909529b8674da6b79f6948939b4bcbb95d.jpg)

Figure 3: Text-to-Image User Study. Percent Win Rate on SDXL measuring image fidelity and image alignment. Significance was assessed with two-sided exact binomial tests on the win and lose counts, excluding ties, with all diferences being highly significant (p < 0.001). OYS significantly outperforms both AYS and default sampling in quality (68.8% win rate vs. AYS, 70.6% vs. Default; ties split equally) and in alignment (62.9% win rate vs. AYS, 66.8% vs. Default).  
![](images/3c10a04bfd4f4e73c9f21c7fcfc700e861f6313f4815803f2ff079e259a9d909.jpg)  
Figure 4: Inpainting Results for SDv1.5 and SDXL against default and OYS schedule. OYS reconstructions remain faithful to the unmasked regions, whereas the 5-step default schedule visibly corrupts the full image.

Inpainting. We evaluate OYS on inpainting. We optimize a 5-step sampling schedule for both Stable Difusion 1.5 and Stable Difusion XL on the training split of COCO Captions (Chen et al., 2015) with a batch size of 64, evaluating on the validation split. For both models, we use the PNDMScheduler (Liu et al., 2022). The optimization objective is LPIPS (Zhang et al., 2018), which measures perceptual similarity between the inpainted sample and the ground truth. The tuned schedules outperform the defaults on both models, improving PSNR by 6.9 dB on SDXL and 2.9 dB on SDv1.5 and winning 93.4% and 85.1% of HPS comparisons, respectively (see Table 3 and Figure 4). Note that these models are trained to alter the full image, not just the masked region, enabling smoother transitions between inpainted and unmasked areas; under the 5-step default schedule this freedom becomes a failure mode — even unmasked regions are corrupted — while the OYS schedule preserves them.

![](images/02a8853d3b755dabf8c11aea39f0f7f39d9d5787d8541b3ba367906acf3b6d76.jpg)

![](images/640f44202d18bd551f20e45e54a282e7440017ad83d8d0196b1b3d2c2cb05743.jpg)

![](images/b2e07affbbda6775f886f6908cef4a31e2124d60ded4e9f95eb2247f68e7726b.jpg)  
Figure 5: 5-step text-to-image sampling schedules in log-SNR space. AYS closely tracks the default schedule’s near-uniform log-SNR spacing, while OYS consistently places more steps at low log-SNR, i.e., high noise levels.

Schedule Analysis. For any difusion model, each timestep maps to a log Signal-to-Noise Ratio (log-SNR), providing a unified representation of noise levels across models (Kingma & Gao, 2023). We plot the schedules in log-SNR space in Figure 5. AYS and Default appear remarkably similar: AYS refines locally from a heuristic initialization with early stopping, which keeps its schedule close to the default. In contrast, OYS allocates more timesteps to higher noise levels, which improves downstream image quality and prompt alignment at low step budgets.

Optimization Eficiency. Figure 6 shows optimized schedule performance as a function of generations used during optimization. Performance for all models saturates before 170K images, with SDXL saturating before 17K generations. Since these are 5-step generations, the SDXL cost is equivalent to approximately 1.7K 50-step generations. In comparison, Sabour et al. (2024) report an upper bound on AYS’s tuning cost—up to 300 iterations of 8,192 samples each, or approximately 2.4 million generations—one to two orders of magnitude more than OYS requires, depending on the model. Even discounting this bound by an order of magnitude, OYS remains cheaper for every model we tune.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td rowspan="2">PSNR↑</td><td rowspan="2">HPS</td><td rowspan="2">HPS</td></tr><tr><td>Average↑ Win Rate↑</td></tr><tr><td rowspan="2">Inverse HED</td><td>Default</td><td>59.79</td><td>0.197</td><td></td></tr><tr><td>OYS</td><td>60.46</td><td>0.206</td><td>65.43%</td></tr><tr><td rowspan="2">Inverse Depth</td><td>Default</td><td>58.51</td><td>0.198</td><td></td></tr><tr><td>OYS</td><td>59.05</td><td>0.202</td><td>57.62%</td></tr><tr><td rowspan="2">Inverse Segment</td><td>Default</td><td>57.97</td><td>0.196</td><td></td></tr><tr><td>OYS</td><td>57.60</td><td>0.201</td><td>58.80%</td></tr></table>

Table 4: Prompt Difusion. PSNR, HPS, and Win Rate on Inverse HED, Inverse Depth, and Inverse Segmentation comparing Default and OYS schedules. HPS Win Rates are with respect to the Default Schedule.

Inverse HED, Inverse Depth, & Inverse Segmentation. We explore sampling optimization for Prompt Difusion (Wang et al., 2023), a model that performs multiple image generation tasks through in-context learning. It accepts both a text prompt, two images demonstrating the desired transformation, and a query image. We optimize schedules for three inverse image tasks: (1) Inverse HED transforms edge maps highlighting object boundaries into photorealistic images; (2) Inverse Depth converts depth maps into visually rich renderings; and (3) Inverse Segmentation transforms semantic segmentation maps into photorealistic images. Using the dataset from Wang et al. (2023), we follow their setup to generate HED, depth, and segmentation maps. We optimize 5-step schedules on the validation split to minimize mean squared error between generated outputs and ground truth images, then evaluate on the test split. Consistent with the prior experiments, OYS achieves higher average HPS on all three tasks, with HPS win rates over the default schedule of 65.4%, 57.6%, and 58.8% on Inverse HED, Inverse Depth, and Inverse Segmentation respectively, and improves PSNR on Inverse HED and Inverse Depth; on Inverse Segmentation PSNR is essentially unchanged (see Table 4 and Figure 8).

![](images/87ab2e2f9cc3677cf2c1434586b273afdebf097c4fe0791dbd0e8f69c43917dd.jpg)

![](images/6c165cceff7f289b4eb7d8fe0ae832c229d8a362f1c8e2548b4e564b473170d3.jpg)  
Figure 6: Optimization eficiency for text-toimage schedule tuning across three difusion models. The graph shows Human Preference Score improvements versus number of generated images during optimization. All models achieve significant HPS gains with relatively few samples, typically plateauing after only 50-100K generations, highlighting OYS’s eficiency relative to methods such as AYS, which can require substantially more generations.  
Figure 7: ImageNet-512 Performance. We compare between the original and OYS-optimized sampling configurations for the EDM2 model family. The graph shows FID scores (lower is better) against model complexity for ImageNet-512. Our OYSoptimized sampling configurations consistently outperform the original sampling configurations from Karras et al. (2024) across all model sizes, from EDM2-XS to EDM2-XXL.

EDM2 Image Generation. We investigate how much performance can be gained from tuning the sampling configurations for the EDM2 family of difusion models (Karras et al., 2024). Karras et al. (2024) adopt the deterministic sampling configurations from Karras et al. (2022), found via a coarse-to-fine grid search. We optimize the parameters of their deterministic sampler discussed in section $2 - \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } .$ , and $\rho - \mathrm { j o i n t l y }$ with the EMA length and guidance strength, using the default guidance network provided by each model, to minimize FID estimated over 8,192 images per configuration.

OYS improves upon the grid-searched configurations of Karras et al. (2024), achieving FID improvements of 3.57% to 7.00% across model sizes (see Figure 7) on ImageNet-512. The optimized configurations depart from the defaults in consistent ways: OYS narrows the sampled noise range at both ends, dropping $\sigma _ { \mathrm { m a x } }$ from its default of 80 to roughly 20 for every model while raising $\sigma _ { \mathrm { m i n } }$ five-fold for all but EDM2-XS, and it increases guidance for the smaller models but disables it entirely for EDM2-L and larger. We provide the full configurations and further analysis in the Appendix.

## 5 Conclusion

We present Optimize Your Sampling (OYS), a framework that casts difusion sampling parameter selection as a black-box optimization problem, tuning the evaluation metric itself rather than a surrogate. It requires no retraining and applies to any pretrained model, sampler, or objective. OYS improves on both AYS and the default schedule for text-to-image generation, and on the default schedule for inpainting and inverse image tasks. The gains extend to SDXL-Turbo, a model already distilled for few-step sampling, showing that schedule tuning remains applicable even after distillation. A 5-step OYS schedule retains 89%–94% of the quality of a 50-step schedule at a tenth of the inference cost, and the improvements largely hold on metrics never used during tuning and are corroborated by human raters — evidence that they reflect genuine quality gains rather than exploitation of the tuning metric.

Inspecting the discovered configurations reveals consistent patterns that prevailing heuristics miss. In log-SNR space, short schedules benefit from allocating more steps to high-noise regions, in contrast to the approximately uniform spacing that both the default and AYS schedules exhibit. For the EDM2 family, where OYS tunes a parametric schedule rather than the timesteps themselves, the optimizer drives $\sigma _ { \mathrm { m a x } }$ far below its default for every model size. Together these suggest that conventional spacing and noise-range heuristics leave real quality on the table at low step budgets, and that the parameters worth tuning extend beyond the timesteps alone.

![](images/94b4109d676dff8fe04e4aafd12aeb20c9c94ecb37dad8c880219362c6dc730b.jpg)  
Figure 8: Prompt Difusion Results for Inverse Depth and Inverse HED against default and OYS schedule. Best viewed in color.

OYS does not aim to produce universal schedules — optimal noise schedules are inherently domain-dependent (Chen, 2023) — but rather to make the already-necessary process of domain-specific tuning simpler and cheaper. Its cost is incurred once per model and task and amortized over all subsequent sampling. OYS thus ofers a practical path to more eficient sampling across architectures and tasks.

## Acknowledgements

JL is supported by a Google PhD Fellowship. This work is also supported by the National Science Foundation through the AI Research Institutes program Award No. DMR-2433348; the National Institute of Food and Agriculture (USDA/NIFA); the Air Force Ofice of Scientific Research (AFOSR); New York-Presbyterian for the NYP-Cornell Cardiovascular AI Collaboration; and a Schmidt AI2050 Senior Fellowship.

## References

Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, and Eytan Bakshy. Unexpected improvements to expected improvement for bayesian optimization. Advances in Neural Information Processing Systems, 36: 20577–20612, 2023. 6

DeepFloyd Lab at StabilityAI. DeepFloyd IF: a novel state-of-the-art open-source text-to-image model with a high degree of photorealism and language understanding. https://www.deepfloyd.ai/deepfloyd-if, 2023. 6

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video difusion: Scaling latent video difusion models to large datasets, 2023. 1

Ting Chen. On the importance of noise scheduling for difusion models. arXiv preprint arXiv:2301.10972, 2023. 11

Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015. 6, 8

Ciprian Corneanu, Raghudeep Gadde, and Aleix M Martinez. LatentPaint: Image inpainting in latent space with difusion models. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 4334–4343, 2024. 1

Roman Garnett. Bayesian Optimization. Cambridge University Press, 2023. 2, 5

Zhengyang Geng, Mingyang Deng, Xingjian Bai, Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. Advances in Neural Information Processing Systems, 38:75460–75482, 2026a. 2

Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J Zico Kolter, and Kaiming He. Improved mean flows: On the challenges of fastforward generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 30467–30476, 2026b. 2

Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. BOOT: Data-free distillation of denoising difusion models with bootstrapping. In ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling, 2023. 2

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. Advances in neural information processing systems, 30, 2017. 6

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 1

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video difusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 1

Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler difusion: 1.5 FID on ImageNet512 with pixel-space difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 18062–18071, June 2025. 8, 15

Yilie Huang, Wenpin Tang, and Xunyu Zhou. ART for difusion sampling: A reinforcement learning approach to timestep schedule, 2026. URL https://arxiv.org/abs/2601.18681. 4

Donald R Jones, Matthias Schonlau, and William J Welch. Eficient global optimization of expensive black-box functions. Journal of Global optimization, 13(4):455–492, 1998. 2, 5

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of difusion-based generative models, 2022. URL https://arxiv.org/abs/2206.00364. 3, 4, 5, 10

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24174–24184, 2024. 3, 5, 10

Diederik Kingma and Ruiqi Gao. Understanding difusion objectives as the ELBO with simple data augmentation. Advances in Neural Information Processing Systems, 36:65484–65516, 2023. 9

Harold J Kushner. A new method of locating the maximum point of an arbitrary multipeak curve in the presence of noise. Journal of Basic Engineering, 1964. 2, 5

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. FLUX.1 Kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742. 6

Benjamin Letham, Brian Karrer, Guilherme Ottoni, and Eytan Bakshy. Constrained bayesian optimization with noisy experiments. Bayesian Analysis, 14(2):495–519, 2019. doi: 10.1214/18-BA1110. 6

Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for difusion models on manifolds. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=PlKWVd2yBkY. 2, 8

Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 2

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE solver for difusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35: 5775–5787, 2022a. 2

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver++: Fast solver for guided sampling of difusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022b. 2, 6

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. Repaint: Inpainting using denoising difusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11461–11471, 2022. 1, 3

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2

Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14297–14306, 2023. 2

Jonas Močkus. On bayesian methods for seeking the extremum. In Optimization Techniques IFIP Technical Conference Novosibirsk, July 1–7, 1974 6, pp. 400–404. Springer, 1975. 2, 5

Michael A Osborne, Roman Garnett, and Stephen J Roberts. Gaussian processes for global optimization. 3rd International Conference on Learning and Intelligent Optimization, pp. 1–15, 2009. 2, 5

William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023. 1

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent difusion models for high-resolution image synthesis. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 1862–1874, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/ 2024/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf. 6

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125, 2022. 1

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022. 1, 6

Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your steps: Optimizing sampling schedules in difusion models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 42947–42975. PMLR, 21–27 Jul 2024. URL https://proceedings. mlr.press/v235/sabour24a.html. 2, 3, 4, 6, 9

Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image difusion models. In ACM SIGGRAPH 2022 conference proceedings, pp. 1–10, 2022a. 1

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image difusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022b. 1

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J. Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4): 4713–4726, 2023. doi: 10.1109/TPAMI.2022.3204461. 1

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of difusion models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=TIdIXIpzhoI. 2

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial difusion distillation. In European Conference on Computer Vision, pp. 87–103. Springer, 2025. 6

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. pmlr, 2015. 1

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising difusion implicit models. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=St1giarCHLP. 1

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 32211–32252. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/song23a.html. 2

Nikita Starodubcev, Denis Kuznedelev, Artem Babenko, and Dmitry Baranchuk. Scale-wise distillation of difusion models. arXiv preprint arXiv:2503.16397, 2025. 2

Antanas Žilinskas. Single-step bayesian search method for an extremum of functions of a single variable. Cybernetics, 11(1):160–166, 1975. 2, 5

Zhendong Wang, Yifan Jiang, Yadong Lu, Pengcheng He, Weizhu Chen, Zhangyang Wang, Mingyuan Zhou, et al. In-context learning unlocked for difusion models. Advances in Neural Information Processing Systems, 36:8542–8562, 2023. 9, 15, 16

Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. DifusionDB: A large-scale prompt gallery dataset for text-to-image generative models. arXiv:2210.14896 [cs], 2022. URL https://arxiv.org/abs/2210.14896. 6

Daniel Watson, Jonathan Ho, Mohammad Norouzi, and William Chan. Learning to eficiently sample from difusion probabilistic models. arXiv preprint arXiv:2106.03802, 2021. 3

Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers for difusion models by diferentiating through sample quality. In International Conference on Learning Representations, 2022. 3

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-Image technical report, 2025. URL https://arxiv.org/abs/2508.02324. 6

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023. 6

Sirui Xie, Zhisheng Xiao, Diederik Kingma, Tingbo Hou, Ying Nian Wu, Kevin P Murphy, Tim Salimans, Ben Poole, and Ruiqi Gao. Em distillation for one-step difusion models. Advances in Neural Information Processing Systems, 37:45073–45104, 2024. 2

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018. 8

Zhenyu Zhou, Defang Chen, Can Wang, Chun Chen, and Siwei Lyu. Simple and fast distillation of difusion models. Advances in Neural Information Processing Systems, 37:40831–40860, 2024. 2

Aihua Zhu, Rui Su, Qinglin Zhao, Li Feng, Meng Shen, and Shibo He. Hierarchical schedule optimization for fast and robust difusion model sampling. Proceedings of the AAAI Conference on Artificial Intelligence, 40(16):13907–13915, Mar. 2026. doi: 10.1609/aaai.v40i16.38400. URL https://ojs.aaai.org/index.php/AAAI/article/view/38400. 4

## A Additional Results

We present detailed results for SDXL-Turbo in Table 5. SDXL-Turbo is a distilled model designed for few-step inference, making it a challenging setting for schedule optimization as the default schedules are already highly compressed. We evaluate OYS with a 3-step Euler Discrete schedule against default schedules at both 1 and 3 steps using the Euler Discrete and Euler Ancestral samplers. OYS achieves the highest mean HPS of 0.298 and consistently outperforms all default baselines, with win rates ranging from 78.55% to 83.94%. Notably, OYS at 3 steps surpasses even the same-step-count defaults by a substantial margin, demonstrating that schedule optimization provides meaningful gains even for models specifically trained for few-step generation.

<table><tr><td>Method</td><td>Sampler</td><td>Steps</td><td>HPS</td><td>OYS vs. Row Average↑ HPS Win Rate↑</td></tr><tr><td>Default</td><td>EA/ED</td><td>1</td><td>0.277</td><td>83.94%</td></tr><tr><td>Default</td><td>ED</td><td>3</td><td>0.287</td><td>83.35%</td></tr><tr><td>Default</td><td>EA</td><td>3</td><td>0.287</td><td>78.55%</td></tr><tr><td>OYS</td><td>ED</td><td>3</td><td>0.298</td><td></td></tr></table>

Table 5: SDXL-Turbo Results. Mean HPS across the COCO Captions validation split and the HPS win rate of OYS (3-step, ED) against each default baseline. ED and EA denote Euler Discrete and Euler Ancestral, respectively.

Hoogeboom et al. (2025) note that zero-shot FID on MSCOCO is known to not necessarily correspond to actual perceived performance. We nevertheless report FID for the text-to-image, inpainting, and Prompt Difusion tasks. Text-to-Image and Inpainting were evaluated on the validation split of COCO Captions (see Table 6 and Table 7), and Prompt Difusion on the test split of the dataset from Wang et al. (2023) (see Table 8).

At 5 steps, OYS substantially improves FID over both the default and AYS schedules on DeepFloyd and SDXL. On SDv1.5, all three 5-step schedules lie within 0.7 FID of one another, with OYS slightly behind the baselines despite winning the HPS comparisons in Table 1 — consistent with the loose coupling between COCO FID and perceived quality noted above.

## B User Study

Figure 9 shows the template used for our user study to evaluate image quality and prompt alignment across diferent sampling schedules. To ensure unbiased assessment, we employed a two-phase evaluation process for each image pair.

In our study methodology, participants first evaluated image quality (Question 1) without considering the prompt. Only after completing this quality assessment were they shown the prompt to evaluate text-image alignment (Question 2). This sequential approach ensured that prompt content did not influence quality judgments.

For each evaluation, we presented participants with pairs of images generated using diferent sampling schedules (Default, AYS, and OYS). The images were randomly shufled and anonymized so participants had no knowledge of which sampling technique generated each image.

The results of this user study, presented in the main body of the paper, demonstrate a strong preference for images generated using our OYS method compared to both Default and AYS sampling schedules in terms of both image quality and prompt alignment.

## C EDM2 Sampling Configurations

We depict the parameters discovered by OYS and compare them with the default parameters in Table 9. The clearest trend is in $\sigma _ { \mathrm { m a x } }$ : OYS drives it down to 20 — the lower limit of our search range — for every model except EDM2-S, which settles just above the limit at 22.1. All are far below the default of 80. That the optimum saturates this bound suggests the preferred $\sigma _ { \mathrm { m a x } }$ may lie lower still, and that the default substantially overestimates the noise level at which sampling needs to begin. $\sigma _ { \mathrm { m i n } }$ shows a complementary trend: every model saturates a limit of its [0.0001, 0.01] search range — EDM2-XS the lower limit, and every other model the upper limit of 0.01, five times the default of 0.002. Together, these results suggest that OYS narrows the sampled noise range at both ends. Guidance splits by model scale: OYS raises it above the default for the smaller models (EDM2-XS through EDM2-M), but for the larger models (EDM2-L through EDM2-XXL) it drives guidance to 1.0 — again the edge of the search range — efectively disabling guidance altogether. The remaining parameters show no consistent trend across models, though we note that ρ also saturates its upper limit of 10 for EDM2-L.

<table><tr><td>Model</td><td>Method</td><td>Steps</td><td>FID ↓</td></tr><tr><td rowspan="6">DeepFloyd</td><td>Default</td><td>50</td><td>27.732</td></tr><tr><td>Default</td><td>10</td><td>25.834</td></tr><tr><td>AYS</td><td>10</td><td>25.496</td></tr><tr><td>Default</td><td>5</td><td>34.912</td></tr><tr><td>AYS</td><td>5</td><td>27.613</td></tr><tr><td>OYS</td><td>5</td><td>25.681</td></tr><tr><td rowspan="5">Stable Diffusion XL</td><td>Default</td><td>50</td><td>25.182</td></tr><tr><td>Default</td><td>10</td><td>26.130</td></tr><tr><td>AYS</td><td>10</td><td>24.584</td></tr><tr><td>Default</td><td>5</td><td>31.264</td></tr><tr><td>AYS</td><td>5</td><td>36.757</td></tr><tr><td rowspan="5">Stable Diffusion v1.5</td><td>OYS</td><td>5</td><td>27.088</td></tr><tr><td>Default</td><td>50</td><td>26.051</td></tr><tr><td>Default</td><td>10</td><td>24.316</td></tr><tr><td>AYS</td><td>10</td><td>24.057</td></tr><tr><td>Default</td><td>5</td><td>24.362</td></tr><tr><td></td><td>AYS</td><td>5</td><td>24.182</td></tr><tr><td></td><td>OYS</td><td>5</td><td>24.865</td></tr></table>

<table><tr><td>Model</td><td>Method</td><td>FID ↓</td></tr><tr><td>SDXL</td><td>Default OYS</td><td>31.80 6.85</td></tr><tr><td>SDv1.5</td><td>Default OYS</td><td>64.83 3.98</td></tr></table>

Table 7: Inpainting FID. FID on the COCO Captions validation split for SDXL and SDv1.5 with 5-step schedules.

Table 6: Text-to-Image FID. FID on the COCO Captions validation split for DeepFloyd, SDXL, and SDv1.5 across schedules and step counts.
<table><tr><td>Task</td><td>Method</td><td>FID↓</td></tr><tr><td>Inverse HED</td><td>Default OYS</td><td>40.93 9.05</td></tr><tr><td>Inverse Depth</td><td>Default OYS</td><td>34.50 12.89</td></tr><tr><td>Inverse Segmentation</td><td>Default OYS</td><td>32.42 13.57</td></tr></table>

Table 8: Prompt Difusion FID. FID on the test split of the dataset from Wang et al. (2023) for the three inverse image tasks with 5-step schedules.

## D Notation

We summarize the notation used throughout the paper in Table 10.

<table><tr><td>Model</td><td> $\sigma _ { \mathrm { m i n } }$ </td><td>σmax</td><td>ρ</td><td>EMA length</td><td>Guidance</td></tr><tr><td colspan="6">EDM2-XS</td></tr><tr><td>Default</td><td>0.002</td><td>80.0</td><td>7.0</td><td>0.045</td><td>1.40</td></tr><tr><td>OYS</td><td>0.0001</td><td>20.0</td><td>4.976</td><td>0.025</td><td>1.758</td></tr><tr><td colspan="6">EDM2-S</td></tr><tr><td>Default</td><td>0.002</td><td>80.0</td><td>7.0</td><td>0.025</td><td>1.40</td></tr><tr><td>OYS</td><td>0.010</td><td>22.132</td><td>9.767</td><td>0.028</td><td>1.659</td></tr><tr><td colspan="6">EDM2-M</td></tr><tr><td>Default</td><td>0.002</td><td>80.0</td><td>7.0</td><td>0.030</td><td>1.20</td></tr><tr><td>OYS</td><td>0.010</td><td>20.0</td><td>3.466</td><td>0.020</td><td>1.505</td></tr><tr><td colspan="6">EDM2-L</td></tr><tr><td>Default</td><td>0.002</td><td>80.0</td><td>7.0</td><td>0.015</td><td>1.20</td></tr><tr><td>OYS</td><td>0.010</td><td>20.0</td><td>10.000</td><td>0.099</td><td>1.000</td></tr><tr><td colspan="6">EDM2-XL</td></tr><tr><td>Default</td><td>0.002</td><td>80.0</td><td>7.0</td><td>0.020</td><td>1.20</td></tr><tr><td>OYS</td><td>0.010</td><td>20.0</td><td>4.098</td><td>0.099</td><td>1.000</td></tr><tr><td colspan="6">EDM2-XXL</td></tr><tr><td>Default</td><td>0.002</td><td>80.0</td><td>7.0</td><td>0.015</td><td>1.20</td></tr><tr><td>OYS</td><td>0.010</td><td>20.0</td><td>4.901</td><td>0.077</td><td>1.000</td></tr></table>

Table 9: EDM2 Sampling Configurations. Comparison of the default and OYS-optimized sampling configurations for each EDM2 model size.

![](images/b77a02efb8d8fbe7b25786b61f6912fce2a5685556a3c59eef82f077c6ec3d77.jpg)  
Figure 9: User Study Template. Instructions and question format used in our user study for evaluating image pairs.

<table><tr><td>Notation</td><td>Definition</td></tr><tr><td colspan="2">Problem setup</td></tr><tr><td>X</td><td>Set of task inputs  $( \mathrm { e . g . }$  , text prompts) used for tuning</td></tr><tr><td>m</td><td>Sampling function of the fixed, pretrained diffusion model</td></tr><tr><td>e</td><td>Black-box evaluation metric</td></tr><tr><td> $\mathbf { p }$ </td><td>Sampling configuration,  $\mathbf { p } = \{ p _ { 1 } , \dots , p _ { M } \}$  ; the parameters OYS optimizes</td></tr><tr><td> $p _ { k }$ </td><td>k-th element of p;  $p _ { k } \in [ 0 , 1 ]$  for the schedule parameters</td></tr><tr><td> $M$ </td><td>Number of parameters in p</td></tr><tr><td> $\mathcal { P }$ </td><td>Search space of sampling configurations</td></tr><tr><td> $\mathbf { p } ^ { * }$ </td><td>Optimal sampling configuration</td></tr><tr><td colspan="2">Sampling schedule</td></tr><tr><td> $K$ </td><td>Number of steps in the sampling schedule</td></tr><tr><td> $s$ </td><td>Function mapping a configuration p to a sampling schedule</td></tr><tr><td> $t _ { \mathrm { m a x } }$ </td><td>Maximum timestep of the diffusion model</td></tr><tr><td> $t _ { k }$ </td><td>k-th timestep of the sampling schedule</td></tr><tr><td> $q$ </td><td>Rounding to the nearest valid integer timestep, for discrete-time models</td></tr><tr><td> $\sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } , \rho$ </td><td>Parameters of the EDM parametric schedule</td></tr><tr><td colspan="2">Bayesian optimization</td></tr><tr><td> $f$ </td><td>Objective function,  $f ( \mathbf { p } ) = e ( m ( \mathcal { X } ; \mathbf { p } ) )$ </td></tr><tr><td> $N$ </td><td>Number of observed evaluations</td></tr><tr><td> $\mathbf { p } ^ { ( j ) }$ </td><td>j-th observed sampling configuration</td></tr><tr><td> $y ^ { ( j ) }$ </td><td>Metric value observed for configuration  $\mathbf { p } ^ { ( j ) }$ </td></tr><tr><td> $\mathcal { D }$ </td><td>Observed configuration-performance pairs,  $( \mathbf { p } ^ { ( j ) } , y ^ { ( j ) } ) _ { j = 1 } ^ { N }$ </td></tr><tr><td colspan="2">Human preference score</td></tr><tr><td> $T$ </td><td>Text prompt</td></tr><tr><td> $I$ </td><td>Generated image</td></tr><tr><td> ${ \mathcal { E } } _ { T }$ </td><td>CLIP text encoder</td></tr><tr><td> $\mathcal { E } _ { I }$ </td><td>CLIP image encoder</td></tr><tr><td> $\omega$ </td><td>Learned scaling parameter of the human preference score</td></tr><tr><td> ${ \mathrm { H P S } } ( T , I )$ </td><td>Human preference score of image I for prompt  $T$ </td></tr><tr><td> $P ( I _ { 1 } \succ I _ { 2 } \mid T )$ </td><td>Predicted preference win rate of image  $I _ { 1 }$  over  $I _ { 2 }$ </td></tr></table>

Table 10: Notation. Summary of the notation used throughout the paper.