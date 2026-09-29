# G<sup>3</sup>-LORA: ORGANIZING REWARD-WEIGHTED VIDEO DATA WITH GRADIENT-GUIDED GROUPED LORA

Jia Song<sup>1,∗</sup> Wenhow Li<sup>1,∗</sup> Lichen Bai<sup>1</sup> Bada Ye<sup>2</sup> Zeke Xie<sup>1,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

jsong231@connect.hkust-gz.edu.cn <sup>†</sup>zekexie@hkust-gz.edu.cn

## ABSTRACT

Post-training foundation video models on heterogeneous reward-weighted data usually assumes that all data categories induce compatible updates. This assumption is fragile when categories correspond to different skills, domains, or evaluation dimensions. We study this problem in text-to-video post-training, where VBench2.0 dimensions define data buckets and an external multimodal reward pipeline assigns sample weights. We propose G<sup>3</sup>-LoRA (Gradient-Guided Grouped LoRA), a data organization procedure that probes category-level gradients induced by rewardweighted video samples, removes the shared global update direction, clusters categories by residual gradient compatibility, trains group-specific LoRA experts, and consolidates them into one adapter by weight merging followed by on-policy distillation from the experts. We motivate this procedure by viewing reward-weighted flow matching as velocity-field regression: incompatible reward dimensions may prefer different denoising directions in overlapping noisy latent regions, causing shared LoRA training to average capabilities. On Wan2.1-T2V-1.3B-Diffusers, the merged grouped adapter improves the matched VBench2.0 evaluation over the base model, a joint reward-weighted LoRA baseline, and random, semantic, and rawgradient partitions trained with the same pipeline; an independent evaluator agrees, and on CogVideoX-2B grouping avoids the negative transfer of joint training. The gain is not uniform: merging compresses the largest specialist gains, distillation recovers part of this loss, and camera motion and several local-quality dimensions remain challenging. Together, these results suggest that gradient compatibility can serve as a practical diagnostic for organizing reward-weighted video post-training data.

## 1 INTRODUCTION

Post-training pipelines increasingly rely on heterogeneous supervision. A single adapter may be trained on data from multiple domains, tasks, preference sources, or quality dimensions. The default engineering choice is simple: mix all data and optimize one model. This choice is convenient, but it hides a strong assumption: data categories that share a model should also share compatible optimization directions.

This paper asks whether that assumption should be measured rather than accepted. We focus on text to-video generation because evaluation already decomposes model quality into multiple dimensions. VBench and VBench2.0 evaluate properties such as human identity, camera motion, material change, multi-view consistency, and complex plot faithfulness (Huang et al., 2024; Zheng et al., 2025). These dimensions are usually treated as reporting axes after generation, but they can also define post-training data buckets. A model can generate candidate videos for each bucket, an external multimodal model can score them, and the resulting reward-weighted data can be used to train a LoRA adapter (Hu et al., 2022). The open question is how those buckets should be combined before adapter training.

The answer is not obvious. Some dimensions appear semantically related but may prefer different local updates; others appear unrelated but may share geometric, temporal, or object-centric structure.

Human identity consistency may reward stable subject appearance, while dynamic attribute tasks emphasize visible change. Camera motion and multi-view consistency may share 3D structure, but they can differ in how much motion the generated video should contain. A single mixed adapter may improve average quality while quietly degrading conflict-heavy dimensions. Conversely, training one adapter per dimension avoids interference but loses data sharing. We therefore treat category organization as a first-class post-training problem.

The key mechanism is specific to diffusion and flow SFT. Sample-level rewards do not directly optimize a scalar preference at inference time; instead, they reweight the regression of the denoising or velocity field. Each evaluation dimension can therefore be viewed as inducing a reward-reweighted conditional velocity field. When multiple dimensions favor incompatible generation behaviors, their preferred velocity directions may conflict in overlapping noisy latent regions. Since the MSE objective learns a conditional mean target, joint training with a shared LoRA can average these directions, producing gradient interference and capability averaging. To address this, we organize data categories by their measured optimization compatibility before training grouped adapters.

Therefore, we propose $_ { \mathrm { G ^ { 3 } - L o R A : } }$ Gradient-Guided Grouped LoRA. The method first computes a reward-weighted gradient for each data category using the same flow-matching loss used in training (Lipman et al., 2023). It then computes a global mixed gradient and removes the component of each category gradient explained by this shared direction. The residual gradients define a compatibility matrix: positive cosine similarity suggests that two categories induce similar category-specific update directions, while negative similarity suggests potential interference. We cluster this matrix into groups, train one LoRA expert per group, and merge the experts into a final adapter. Because merging can dilute group-specific updates, we then distill the experts into the merged adapter on its own sampling trajectories (on-policy distillation, OPD) (Agarwal et al., 2024; Fang et al., 2026).

The concrete instantiation uses Wan2.1-T2V-1.3B-Diffusers (WanTeam et al., 2025; Wan-AI, 2025; Hugging Face, 2024), 17 non-diversity VBench2.0 dimensions, and Qwen3-VL-8B-Instruct-based QA rewards (Bai et al., 2025b); we repeat the procedure on CogVideoX-2B (Yang et al., 2025), which differs in architecture and training objective. The broader point is not specific to this benchmark. VBench2.0 provides a convenient taxonomy and evaluator, but the proposed procedure applies whenever a post-training dataset can be partitioned into reward-bearing categories whose interactions are uncertain.

## Contributions.

• We formulate heterogeneous video post-training as a data organization problem: categories should be grouped according to optimization compatibility, not only semantic labels.

• We connect reward-weighted flow SFT to reward-reweighted velocity-field regression, providing a local mechanism for capability averaging under shared LoRA training.

• We introduce a residual gradient compatibility measure that subtracts the global mixed update direction before comparing category-specific gradients, and show that the conflict it reveals is reproducible across probe sets and backbones and predicts held-out transfer.

• We instantiate the method on two video backbones with matched grouping controls, an independent evaluator, and paired bootstrap intervals, and we locate where merging loses specialist capability and how much on-policy distillation recovers.

## 2 METHOD

## 2.1 PROBLEM SETTING

Let $\mathcal { D } = \{ \mathcal { D } _ { 1 } , . . . , \mathcal { D } _ { K } \}$ be a post-training dataset partitioned into K categories. In our implementation, $K = 1 7$ and each category corresponds to a non-diversity VBench2.0 dimension. Training videos are generated by the same base model that will later be post-trained, so the dataset is a self-generated improvement pool rather than a separately collected human-video corpus. A sample ${ \boldsymbol { x } } = ( v , p , c , w )$ contains a generated video v, prompt p, category $c ,$ and scalar reward weight w. The reward comes from a dimension-specific QA pipeline: a multimodal model observes the video, answers generated questions, and the fraction of expected answers is converted into a sample-level training weight. The weight is used as a scalar multiplier on the flow-matching loss; it is not an RL objective and does not require preference-pair optimization.

![](images/8ee8e1a151aa0274db6a20e3a10bae5c607fe2d935e394a6a763aee562364c36.jpg)  
Figure 1: Method overview. $\mathbf { G } ^ { 3 } \mathbf { - } \mathbf { L o R A }$ treats post-training categories as optimization objects. It uses reward-weighted category gradients to estimate compatibility, clusters categories after removing the global mixed direction, trains group-specific LoRA experts, merges them, and distills the experts back into the merged adapter.

We train only LoRA parameters $\phi$ on top of a frozen video generation model. For each video, the VAE encodes frames into a latent $z _ { 1 }$ , noise $z _ { 0 } \sim \mathcal { N } ( 0 , I )$ is sampled, and a time t defines the flow-matching interpolation as

$$
z _ { t } = ( 1 - t ) z _ { 0 } + t z _ { 1 } , \qquad u = z _ { 1 } - z _ { 0 } .\tag{1}
$$

The model predicts $\hat { u } _ { \phi } ( z _ { t } , p , t )$ and optimizes a reward-weighted loss as

$$
\mathcal { L } ( \phi ; x ) = \frac { w } { \bar { w } } \left. \hat { u } _ { \phi } ( z _ { t } , p , t ) - u \right. _ { 2 } ^ { 2 } ,\tag{2}
$$

where w¯ is the global mean reward weight. Reward weighting is not only sample filtering: it changes which videos dominate the category gradient and therefore changes the compatibility structure that $\mathrm { G ^ { 3 } }$ -LoRA estimates. The naive joint baseline minimizes the average of $\operatorname { E q }$ . 2 across all categories. $\mathrm { G ^ { 3 } } .$ -LoRA instead uses gradients of Eq. 2 to decide which categories should share an adapter.

Capability averaging under MSE regression. For a fixed prompt and nearby noisy latent region, suppose two reward dimensions emphasize different target velocities $u _ { 1 }$ and $u _ { 2 }$ . A shared MSE regressor trained on both distributions prefers a conditional mean direction rather than either specialized direction. In video generation, this can manifest as softened camera motion, weaker attribute change, or partial preservation of structure without fully satisfying the corresponding dimension. We use this mechanism as the local explanation for why reward-weighted post-training can improve the mean score while still degrading conflict-heavy dimensions.

## 2.2 GRADIENT-GUIDED GROUPED LORA

Reward-weighted category probing. For each category i, we estimate a reward-weighted LoRA gradient as

$$
g _ { i } = \frac { 1 } { | B _ { i } | } \sum _ { x \in B _ { i } } \nabla _ { \phi } \mathcal { L } ( \phi ; x ) ,\tag{3}
$$

where $B _ { i }$ is a fixed probe subset. To reduce probe noise, the implementation fixes diffusion timesteps and noise seeds during probing. Probing is repeated at multiple training stages, e.g., 0%, 25%, 50%, $7 5 \% ,$ , and 100% of a reference joint training trajectory. The resulting gradient is a tractable proxy for the reward-reweighted velocity field induced by a category.

Removing the shared update direction. Raw category gradients can be dominated by a shared update that is useful for all categories, such as adapting the model to the video resolution, prompt distribution, or LoRA parameterization. Comparing raw gradients can therefore overstate compatibility because the common direction can mask category-specific conflicts. We compute a global gradient $g _ { \mathrm { g l o b a l } }$ on a mixed probe batch and remove its projection from each category gradient as

$$
r _ { i } = g _ { i } - \frac { \langle g _ { i } , g _ { \mathrm { g l o b a l } } \rangle } { \| g _ { \mathrm { g l o b a l } } \| _ { 2 } ^ { 2 } + \epsilon } g _ { \mathrm { g l o b a l } } .\tag{4}
$$

The residual $r _ { i }$ captures category-specific pressure after accounting for the common direction. Directly subtracting $g _ { \mathrm { g l o b a l } }$ instead gives the same partition (adjusted Rand index 1.0; Appendix E), so the essential operation is removing the shared direction rather than the specific projection operator.

Compatibility and grouping. The residual compatibility between categories i and $j$ is defined as

$$
S _ { i j } = \frac { \left. { { r } _ { i } } , { { r } _ { j } } \right. } { \left\| { { r } _ { i } } \right\| _ { 2 } \left\| { { r } _ { j } } \right\| _ { 2 } + \epsilon } .\tag{5}
$$

When multiple probe stages are available, we average $S _ { i j }$ across stages to obtain $\bar { S } _ { i j }$ . We then run average-linkage hierarchical clustering with distance $D _ { i j } = 1 - \bar { S } _ { i j }$ and select M groups. The experiments use $M = 5$

Grouped LoRA training and merge. For each group $G _ { m }$ , we train an independent LoRA expert using only samples from categories in $G _ { m }$ and the reward-weighted objective in Eq. 2. Group-specific training prevents strongly conflicting categories from sharing all update steps, while still allowing compatible categories to share data. The final adapter is obtained by merging group experts into a single deployable LoRA module with a dense delta-SVD procedure: group LoRA updates are expanded to dense weight deltas, averaged with group-size weights, and recompressed into one LoRA adapter. Appendix F gives the dense merge and recomposition equations. This can be viewed as a weight-space approximation to an ensemble or mixture of skill-specific experts: training creates several local specialists, while merging folds them into one adapter for inference.

Merge retention. The success of grouped training depends on whether the learned LoRA updates remain compatible after merging. If two group deltas satisfy $\Delta W _ { i } \approx - \Delta W _ { j }$ , uniform averaging can cancel both capabilities. Even when parameter-space cosine is small, the updates may still conflict on real activations. We therefore treat merge retention as a diagnostic: a merged adapter should preserve a substantial fraction of each group expert’s gain on the dimensions inside that group.

On-policy expert distillation (OPD). When merging loses part of an expert’s gain, we fine-tune the merged adapter $\phi _ { 0 }$ so that, for each prompt, its velocity matches the expert $\psi _ { m ( p ) }$ of the prompt’s group on latent states $z _ { \tau }$ drawn from the student’s own sampling trajectories $\rho _ { \phi } ( \cdot \mid p )$ . Initializing $\phi  \phi _ { 0 }$ , OPD minimizes

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { O P D } } ( \phi ) = \mathbb { E } _ { p , \ z _ { \tau } \sim \rho _ { \phi } ( \cdot \vert p ) } \Big [ \big \| \hat { u } _ { \phi } ( z _ { \tau } , p , \tau ) - \mathrm { s g } \ \hat { u } _ { \psi _ { m ( p ) } } ( z _ { \tau } , p , \tau ) \big \| _ { 2 } ^ { 2 } } \\ & { \qquad + \lambda \big \| \hat { u } _ { \phi } ( z _ { \tau } , p , \tau ) - \mathrm { s g } \ \hat { u } _ { \phi _ { 0 } } ( z _ { \tau } , p , \tau ) \big \| _ { 2 } ^ { 2 } \Big ] , } \end{array}\tag{6}
$$

where $\mathrm { s g }$ is stop-gradient and the second term anchors the student to the merged adapter. Sampling states from the student, rather than noising training videos, supervises it on the latents it visits at inference (Agarwal et al., 2024; Fang et al., 2026). The loss uses neither reward weights nor training videos (Appendix H).

## 3 EXPERIMENTS

## 3.1 EXPERIMENTAL SETUP

Model and data. The base model is Wan2.1-T2V-1.3B-Diffusers. Training videos are generated by the same base model from dimension-specific prompts. Each of 17 non-diversity VBench2.0 dimensions contributes approximately 560 videos. Qwen3-VL-8B-Instruct provides reward weights through the dimension-specific QA pipeline; Appendix B describes this reward construction. As a second backbone and objective, we rerun the full procedure on CogVideoX-2B with its native diffusion-denoising loss and rank-128 experts merged exactly into a rank-640 adapter (Appendix I).

Evaluation. We evaluate on VBench2.0 dimensions excluding diversity. All main comparisons use a matched evaluation protocol with aligned prompts, seeds, generation parameters, and evaluator settings; Appendix A gives the concrete generation and evaluation details. Uncertainty is estimated by a paired bootstrap over prompts (Appendix J). As an evaluator independent of the Qwen reward model, we also score the same videos with VideoScore-v1.1 (He et al., 2024). The retention study in Appendix G is a separate matched diagnostic evaluation and is reported only for analyzing merge behavior.

Baselines and compared adapters. The main comparison includes Base Wan2.1 without posttraining; Joint LoRA trained on all 17 reward-weighted dimensions; four M = 5 partition controls, namely two random partitions, the official VBench2.0 semantic categories, and clustering of raw (unprojected) gradients; the merged Grouped LoRA produced by the dense delta-SVD-style merge; and Grouped LoRA + OPD. The partition controls use the same grouped-LoRA training and merge pipeline as G<sup>3</sup>-LoRA, but their dimension partitions are not selected by residual-gradient compatibility; Appendix C lists all partitions. A separate retention evaluation compares the base model, the merged grouped adapter, and each group-only LoRA evaluated only on the VBench2.0 dimensions assigned to that group.

## 3.2 GRADIENT GROUPING ANALYSIS

The gradient analysis produces five groups over the 17 non-diversity VBench2.0 dimensions:

• Group 1: dynamic spatial relationship, human clothes consistency, instance preservation, motion order.

• Group 2: human anatomy, human identity consistency, human interaction, motion rationality.

• Group 3: camera motion, complex landscape, composition.

• Group 4: dynamic attribute, material, multi-view consistency, thermotics.

• Group 5: complex plot, mechanics.

Figure 2 visualizes the matrix used for grouping. The strongest positive pairs are not arbitrary semantic neighbors: material and thermotics have mean similarity 0.841, dynamic attribute and thermotics have 0.824, and human anatomy and human identity consistency have 0.709. Conversely, the stable negative pairs reveal conflicts that a naive semantic grouping could miss, such as human clothes consistency versus thermotics (-0.818), dynamic attribute versus human clothes consistency (- 0.798), and motion order versus multi-view consistency (-0.630). These values are averaged over five probe stages, making the matrix a staged compatibility estimate rather than a single-batch diagnostic.

We choose M = 5 because it gives a cleaner residual-gradient partition than M = 4. Under averagelinkage clustering on $\boldsymbol { D } = \boldsymbol { 1 } ^ { \intercal } - \boldsymbol { \bar { S } }$ , the $M = 5$ split increases within-group mean similarity from 0.345 to 0.473, improves the within-minus-between separation from 0.559 to 0.633, and removes all negative within-group pairs. In contrast, the M = 4 split leaves three negative within-group pairs, including one strongly negative pair. The grouping is therefore selected to reduce within-adapter gradient conflict rather than to match a hand-written semantic taxonomy.

This grouping has two uses in the paper. First, it is a diagnostic result: it shows that reward-weighted category gradients contain structure that is not visible from metric names alone. Second, it defines the grouped-LoRA training plan: each group trains one LoRA expert, and the experts are merged before VBench2.0 evaluation. The downstream results below show that the grouping improves the average score and reduces the worst dimension-level drop, but also exposes a dimension-level tradeoff: camera motion does not inherit the improvement obtained by joint LoRA. Appendix E provides the corresponding dendrogram and stage-wise heatmaps.

This conflict is hidden by the shared direction and is not specific to one backbone. With raw gradients, all 136 category pairs on Wan2.1 have positive cosine similarity; after removal, 75 pairs (55%) are negative, and on CogVideoX-2B 81–87 pairs are negative across three non-overlapping probe cohorts. Resampled probe sets reproduce the Wan2.1 matrix (Pearson 0.847), and residual similarity predicts how a step along one category’s gradient changes another category’s held-out loss (Spearman $\rho = 0 . 2 5 6$ , permutation $p = 0 . 0 0 3 )$ .

![](images/b06ee13fcd6c8f0ede30738f77cca80e2a9701cdb2c49efac11054543ddd777f.jpg)  
Figure 2: Residual gradient compatibility across VBench2.0 dimensions. Each entry is the mean cosine similarity between two category gradients after projecting out the global mixed-gradient direction. Dimensions are ordered by the final $M = 5$ grouping, so block structure indicates compatible categories and blue off-block regions indicate potential interference.

Table 1: VBench2.0 comparison on Wan2.1. Grouped LoRA (Ours) obtains the best mean among merged adapters and a smaller worst drop than joint LoRA or the random partitions; OPD raises the mean further. Ties count as neither improved nor degraded; “–”: not computed.
<table><tr><td>Method</td><td>Mean</td><td>∆ Mean</td><td>Improved / degraded dims</td><td>Worst drop vs. Base</td></tr><tr><td>Base Wan2.1</td><td>47.07</td><td></td><td></td><td></td></tr><tr><td>Joint LoRA</td><td>46.99</td><td>-0.07</td><td>9/8</td><td>-8.00</td></tr><tr><td>Random partition A</td><td>46.38</td><td>-0.69</td><td>6/9</td><td>-8.21</td></tr><tr><td>Random partition B</td><td>45.82</td><td>-1.25</td><td>8/8</td><td>-16.00</td></tr><tr><td>Semantic partition</td><td>47.20</td><td>+0.13</td><td></td><td></td></tr><tr><td>Raw-gradient partition</td><td>47.35</td><td>+0.28</td><td></td><td></td></tr><tr><td>Grouped LoRA (Ours)</td><td>48.18</td><td>+1.11</td><td>9/6</td><td>-4.52</td></tr><tr><td>Grouped LoRA + OPD (Ours)</td><td>49.91</td><td>+2.84</td><td>12 /4</td><td>-3.97</td></tr></table>

## 3.3 VBENCH2.0 RESULTS

Table 1 gives the downstream picture. Joint LoRA is slightly below the base model in mean score (- 0.07), despite producing gains on camera motion (+8.00), mechanics (+5.18), multi-view consistency (+4.64), and dynamic spatial relationship (+4.00). This pattern is consistent with a single mixed adapter under heterogeneous reward weighting: some dimensions benefit, but the adapter also pays for the mixture with regressions in human interaction (-8.00), material (-7.63), and dynamic attribute (-6.00).

The two random partitions are also weak: Random partition A reaches 46.38 mean score, while Random partition B reaches 45.82. Both are below the base model, joint LoRA, and Grouped LoRA. Random partition A improves 6 out of 17 dimensions with a worst dimension-level drop of -8.21, and Random partition B improves 8 dimensions but has a larger worst drop of -16.00. The semantic and raw-gradient partitions are only marginally above the base model (47.20 and 47.35); the semantic partition places 13 negatively related pairs in the same group, whereas the residual partition places none. These comparisons indicate that the benefit of the grouped adapter is not explained by simply splitting the 17 dimensions into several smaller LoRA experts; the partition itself matters.

![](images/2fbf88ef0cf047ef0f015a8724f3cd5cee284c09e6628090124123e290d48b67.jpg)  
Figure 3: Per-dimension score changes over the base model. Grouped LoRA (Ours) improves multi-view consistency and motion order, while material, human interaction, and camera motion reveal remaining dimension-specific tradeoffs. <sup>†</sup>Group-only experts are a non-deployable diagnostic: each dimension is scored with the expert of its own group. Colors use a fixed range of −20 to +20; the upper colorbar extension indicates values above +20. Annotations show score changes rounded to one decimal place; the last column is the mean change across 17 dimensions.

Grouped LoRA increases the mean to 48.18, a +1.11 gain over the base model and a +1.19 gain over joint LoRA. The largest gains over the base model occur on multi-view consistency (+8.99), motion order understanding (+8.00), dynamic spatial relationship (+4.00), thermotics (+3.29), and complex landscape (+2.40). These improvements support the main hypothesis that grouping can reduce some forms of capability averaging: the grouped adapter improves the average tradeoff and substantially reduces the worst dimension-level drop compared with joint LoRA. The gain over the base model is positive in 96.6% of paired bootstrap resamples (95% CI [−0.08, +2.26]), so it is a consistent trend rather than a significant difference at the 5% level.

At the same time, the result is not a clean dominance claim. The grouped adapter still degrades material (-4.52), human interaction (-4.00), and several identity or clothing dimensions relative to the base model. It also gives up the joint LoRA’s camera-motion gain: camera motion remains at the base score of 12.00, while joint LoRA reaches 20.00. This is the clearest diagnostic contrast in Appendix Table 8. VBench2.0 camera motion is a hard label match over CoTracker-estimated motion classes, so improving the mean score and multi-view consistency does not guarantee better camera-trajectory control.

Figure 3 summarizes these dimension-level tradeoffs. Appendix Table 8 reports the full dimensionlevel table, including the appendix-only rank-8 merge diagnostic and the OPD adapter.

Figure 4 illustrates how these dimension-level differences can appear in generated videos. We choose one Composition prompt that requires a dog, a cat, and a basket to remain correctly bound over time, and one Thermotics prompt that requires a visible physical state change under heating. These examples target two different failure modes: relational binding among multiple entities and temporal transformation of material state.

In the Composition example, the base model produces the requested objects but the basket relation is mostly static. Grouped LoRA (Ours) keeps the dog, cat, and basket relation more coherent across the shown frames. In the Thermotics example, the base output remains close to a static cheese block, whereas Grouped LoRA (Ours) shows clearer gradual deformation, matching the intended heat-driven transformation. The cases therefore provide visual evidence for the same pattern seen in the automatic metrics: grouping is most helpful when the target dimension requires preserving a specific relation or transformation rather than merely increasing generic visual quality. They are single cases, however: over all Composition prompts, the score is essentially unchanged (47.80 versus 48.10).

![](images/5427ed22565916f46856bb13830e86910e8eaa6417a6908ebb50bcd24acb819b.jpg)  
Figure 4: Qualitative comparison on representative VBench2.0 dimensions. We compare Base Wan2.1 and Grouped LoRA (Ours) on Composition and Thermotics prompts; each row shows five temporal frames from one generated video. In the Composition case, Grouped LoRA more consistently preserves the relation among the dog, cat, and basket. In the Thermotics case, it shows clearer signs of cheese softening or melting. These examples are qualitative case studies and complement, rather than replace, the automatic VBench2.0 scores.

Table 2: Second backbone and independent evaluator. Left: CogVideoX-2B, VBench2.0 macro score over 17 dimensions (×100). Right: VideoScore-v1.1 five-aspect mean on the same Wan2.1 videos used in Table 1. Brackets: paired 95% bootstrap CI.
<table><tr><td colspan="2">CogVideoX-2B (VBench2.0)</td><td colspan="2">Wan2.1 (VideoScore-v1.1)</td></tr><tr><td>Base</td><td>39.87</td><td>Base</td><td>2.914</td></tr><tr><td>Joint LoRA Grouped LoRA (Ours)</td><td>27.95 40.99</td><td>Grouped LoRA (Ours)</td><td>3.080</td></tr><tr><td colspan="2">Ours – Joint +13.04 [9.08, 16.92]</td><td>Ours — Base</td><td>+0.166 [0.140, 0.192]</td></tr></table>

## 3.4 SECOND BACKBONE AND INDEPENDENT EVALUATOR

On CogVideoX-2B, which changes both the architecture and the training objective, joint training degrades the base model from 39.87 to 27.95, whereas the grouped adapter reaches 40.99 (Table 2). Most of this margin reflects avoided negative transfer rather than a large gain over the base model (+1.12). The joint degradation is already present at 25% of training (36.78), so checkpoint selection does not rescue it, and it is dimension-selective: human anatomy, human identity, and thermotics improve while human interaction, motion order, and motion rationality collapse (Appendix I), consistent with capability averaging rather than a globally failed run.

Because the training rewards come from Qwen3-VL, we rescored the Wan2.1 videos with VideoScore v1.1, a learned metric trained on human ratings of generated videos and not built on Qwen. Grouped LoRA improves all five VideoScore aspects over the base model (Table 2; Appendix J).

## 3.5 DIMENSION-LEVEL DIAGNOSTICS

To assess whether merging contributes to the remaining dimension-level tradeoffs, we compare each group expert with the merged adapter on its assigned dimensions.

In the matched retention evaluation (Appendix G; group-only row of Figure 3), group-only LoRAs reach 48.06 on their assigned dimensions, compared with 48.18 for the merged grouped adapter and 47.07 for the base model. The small average merge gap suggests no global capability collapse, while the larger gaps on multi-view consistency and mechanics indicate that weight-space merging is a plausible source of remaining group-specific tradeoffs. On all six dimensions where the expert beats the base model, the merged adapter is below the expert; on the other eleven, it is above the expert, indicating transfer across groups. Selecting post hoc the better of the two for each dimension gives a diagnostic upper bound of 51.07. Neither rank truncation (error $1 . 5 \times 1 0 ^ { - 6 } )$ nor sign cancellation between expert deltas (no negative pair in 240 modules) explains this loss (Appendix G).

Distilling the experts back into the merged adapter (Eq. 6) raises the mean from 48.18 to 49.91 (+1.73; paired 95% bootstrap CI [+0.42, +3.05]), closing about 60% of the gap to this upper bound, and improves 12 of 17 dimensions over the base model (Table 1; last row of Figure 3). The largest gains are on multi-view consistency (+8.07), human interaction (+6.00), and complex plot (+5.07). Human interaction reaches 68.00 although its expert scores 58.00, so OPD does not simply copy the teacher. Recovery is uneven: only 0.71 of the 15.14-point mechanics gap is recovered, composition decreases slightly (−0.50), and camera motion is unchanged. The interval reflects evaluation sampling for one OPD training run; variation across training runs was not measured.

## 4 RELATED WORK

Text-to-video generation builds on diffusion and flow-based modeling (Ho et al., 2020; Rombach et al., 2022; Peebles & Xie, 2023; Blattmann et al., 2023), with evaluation moving toward multi-axis diagnosis through EvalCrafter, T2V-CompBench, VideoPhy, VBench, and VBench2.0 (Liu et al., 2023; Sun et al., 2024; Bansal et al., 2024; Huang et al., 2024; Zheng et al., 2025). We use VBench2.0 dimensions as both metrics and data categories. Diffusion alignment and efficient video generation are reviewed by Liu et al. (2026) and Shao et al. (2026), respectively. Zigzag Diffusion Sampling (Bai et al., 2025a) and Weak-to-Strong Diffusion with Reflection (Bai et al., 2026a) improve generation through reflective sampling without retraining.

Our training uses LoRA (Hu et al., 2022) and Qwen3-VL-8B-Instruct QA rewards (Bai et al., 2025b), but differs from RLHF/DPO-style language alignment (Bai et al., 2022; Rafailov et al., 2023) and diffusion reward optimization methods (Black et al., 2023; Wallace et al., 2024; Prabhudesai et al., 2023): the goal is not a new objective, but organizing reward-weighted video data. CRAFT (Sun et al., 2026) studies composite-reward filtering followed by supervised fine-tuning; our focus is instead on which reward-weighted categories should share updates.

The grouping signal is related to gradient conflict and multi-task optimization methods (Sener & Koltun, 2018; Chen et al., 2018; Yu et al., 2020; Wang et al., 2021; Liu et al., 2021), while the merged adapter connects to weight-space merging (Wortsman et al., 2022; Matena & Raffel, 2022; Ilharco et al., 2023; Yadav et al., 2023). G<sup>3</sup>-LoRA intervenes between these lines: it probes residual category gradients before training specialists, then merges the resulting LoRA experts into one deployable adapter. CASA (Wang et al., 2026) addresses spectral interference when transferring LoRAs to distilled video backbones, whereas we group and merge experts trained on the same backbone.

Our consolidation follows on-policy distillation (Agarwal et al., 2024) and the multi-teacher paradigm of Flow-OPD (Fang et al., 2026). Our distinction lies in constructing teachers by residualgradient grouping of reward-weighted video categories, rather than by single-reward specialization. MaineCoon (Bai et al., 2026b) also consolidates domain-specialized LoRA experts, using preference optimization and reinforced on-policy distillation for streaming audio-visual generation rather than residual-gradient grouping. Our capability-consolidation goal differs from the few-step generation objective of Adaptive Matching Distillation (Bai et al., 2026c).

## 5 LIMITATIONS

This study covers two open text-to-video backbones of at most 2B parameters and a VBench2.0- derived category taxonomy; on CogVideoX-2B the gain over the base model is small, so the most natural next step is to test whether the same grouping signal appears across larger model families, reward sources, and evaluation suites. Rewards come from Qwen3-VL and VBench2.0 also uses vision-language judgments; VideoScore agreement reduces but does not remove this coupling, and no human study was run. On Wan2.1, the 95% interval of the merged adapter’s gain over the base model includes zero. OPD adds about 205 GPU-hours and was not applied to the control partitions or to CogVideoX. Our gradient compatibility measure is also intentionally lightweight: it is designed as a practical probe for organizing post-training data rather than as a complete account of all denoising dynamics. Future work could combine this probe with richer merge rules, repeated control partitions, and human preference studies to further separate data-organization effects from evaluator- or merge-specific effects.

## 6 CONCLUSION

We presented $\mathrm { G ^ { 3 } }$ -LoRA, a gradient-guided procedure for organizing heterogeneous reward-weighted post-training data. The method computes category gradients, removes the shared global direction, clusters residual compatibility, trains group-specific LoRA experts, and consolidates them by merging and on-policy distillation. The central view is that reward-weighted video SFT induces multiple conditional velocity fields, and post-training should ask not only which samples are high quality, but also which reward-weighted samples should share update directions. VBench2.0 results support this view in a qualified way: Grouped LoRA improves the mean score over both the base model and joint LoRA, while reducing the worst dimension-level drop; it avoids joint training’s negative transfer on CogVideoX-2B, and distillation recovers part of the merging loss. Overall, the results position gradient compatibility as a practical organizing signal for reward-weighted video post-training.

## ETHICS STATEMENT

The method is a post-training data organization procedure and does not directly introduce a new video generation backbone. Its positive impact is that it may make post-training more data-efficient and more diagnosable by exposing when training categories interfere. Its negative impact follows the underlying model class: improved video generation can also improve misleading or harmful generated media. Any future release of trained adapters or generated videos should therefore follow the safety and licensing constraints of the base model and benchmark assets, and should document whether generated samples are filtered before release.

## REPRODUCIBILITY STATEMENT

Section 2 and the appendices describe the method, experimental settings, reward construction, and diagnostic protocols. All methods share a prompt manifest and seed schedule for evaluation. We plan to release the implementation, configurations, partitions, prompt manifests, and raw evaluator outputs.

## AI USE STATEMENT

Generative AI tools were used to assist with writing and formatting. Their use for data generation, reward construction, distillation, and evaluation is described in the main text and appendices. The authors take responsibility for all claims, results, and artifacts.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024.

Lichen Bai, Shitong Shao, Zikai Zhou, Zipeng Qi, Zhiqiang Xu, Haoyi Xiong, and Zeke Xie. Zigzag diffusion sampling: Diffusion models can self-improve via self-reflection. In International Conference on Learning Representations, 2025a. URL https://openreview.net/forum? id=MKvQH1ekeY.

Lichen Bai, Masashi Sugiyama, and Zeke Xie. Weak-to-strong diffusion with reflection. In International Conference on Learning Representations, 2026a. URL https://openreview.net/ forum?id=tg19FVh3p1.

Lichen Bai, Tianhao Zhang, Shitong Shao, Dingwei Tan, Qiyu Zhong, Zhengpeng Xie, Haopeng Li, Qinghao Huang, Dandan Shen, Tengjiao Ji, Wei Wang, Peicheng Wu, Yuxuan Zhao, Xiangyu Zhu, Welly Luo, Shurui Yang, and Zeke Xie. MaineCoon: Pursuing a real-time audio-visual social world model. arXiv preprint arXiv:2606.17800, 2026b. URL https://arxiv.org/abs/ 2606.17800.

Lichen Bai, Zikai Zhou, Shitong Shao, Wenliang Zhong, Shuo Yang, Shuo Chen, Bojun Chen, and Zeke Xie. Optimizing few-step generation with adaptive matching distillation. arXiv preprint arXiv:2602.07345, 2026c. URL https://arxiv.org/abs/2602.07345.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025b.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. arXiv preprint arXiv:2406.03520, 2024.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22563–22575, 2023. doi: 10.1109/CVPR52729.2023.02161.

Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 794–803. PMLR, 2018.

Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, and Feng Zhao. Flow-OPD: On-policy distillation for flow matching models. arXiv preprint arXiv:2605.08063, 2026.

Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, Kai Wang, Quy Duc Do, Yuansheng Ni, Bohan Lyu, Yaswanth Narsupalli, Rongqi Fan, Zhiheng Lyu, Bill Yuchen Lin, and Wenhu Chen. VideoScore: Building automatic metrics to simulate fine-grained human feedback for video generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024. doi: 10.1109/CVPR52733.2024.02060.

Hugging Face. Diffusers: State-of-the-art diffusion models. https://github.com/ huggingface/diffusers, 2024. Software library.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=6t0Kwf8-jrj.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. In Advances in Neural Information Processing Systems, volume 34, pp. 18878–18890, 2021.

Buhua Liu, Shitong Shao, Bao Li, Lichen Bai, Zhiqiang Xu, Haoyi Xiong, James Tin Yau Kwok, Sumi Helal, and Zeke Xie. Alignment of diffusion models: Fundamentals, challenges, and future. ACM Computing Surveys, 58(9):1–37, 2026. doi: 10.1145/3796982. Article 244.

Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023.

Michael Matena and Colin Raffel. Merging models with fisher-weighted averaging. In Advances in Neural Information Processing Systems, volume 35, pp. 17703–17716, 2022.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023. doi: 10.1109/ICCV51070.2023.00387.

Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10674–10685, 2022. doi: 10.1109/CVPR52688.2022.01042.

Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems, volume 31, pp. 525–536, 2018.

Shitong Shao, Lichen Bai, Pengfei Wan, James Kwok, and Zeke Xie. Efficient video diffusion models: Advancements and challenges. arXiv preprint arXiv:2604.15911, 2026. URL https: //arxiv.org/abs/2604.15911.

Kaiyue Sun, Kaiyi Huang, Xian Liu, Yue Wu, Zihan Xu, Zhenguo Li, and Xihui Liu. T2vcompbench: A comprehensive benchmark for compositional text-to-video generation. arXiv preprint arXiv:2407.14505, 2024.

Zening Sun, Zhengpeng Xie, Lichen Bai, Shitong Shao, Shuo Yang, and Zeke Xie. CRAFT: Aligning diffusion models with fine-tuning is easier than you think. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. URL https://arxiv.org/ abs/2603.18991.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024. doi: 10.1109/CVPR52733.2024.00786.

Wan-AI. Wan2.1-t2v-1.3b-diffusers. https://huggingface.co/Wan-AI/Wan2. 1-T2V-1.3B-Diffusers, 2025. Hugging Face model card.

Yuchen Wang, Wenliang Zhong, Lichen Bai, Zikai Zhou, Shitong Shao, Bojun Cheng, Shuo Chen, Shuo Yang, and Zeke Xie. Exploring data-free LoRA transferability for video diffusion models. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/ 2605.01929.

Zirui Wang, Yulia Tsvetkov, Orhan Firat, and Yuan Cao. Gradient vaccine: Investigating and improving multi-task optimization in massively multilingual models. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= F1vEjWK-lH\_.

WanTeam, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. doi: 10.48550/arXiv.2503.20314.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pp. 23965–23998. PMLR, 2022.

Prateek Yadav, Derek Tam, Leshem Choshen, Colin A. Raffel, and Mohit Bansal. TIES-merging: Resolving interference when merging models. In Advances in Neural Information Processing Systems, volume 36, 2023.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, and Jie Tang. CogVideoX: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. In Advances in Neural Information Processing Systems, volume 33, pp. 5824–5836, 2020.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, et al. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

## A TRAINING, DISTILLATION, AND EVALUATION PARAMETERS

The main LoRA comparisons use a quick VBench2.0 protocol with 25 prompts per dimension and two generated videos per prompt. This matched protocol is used to compare adapters under identical generation and evaluator settings while keeping evaluation cost practical; the same quick protocol is used consistently for the base, joint, partition-control, grouped, and OPD comparisons.

Table 3: Generation parameters for the quick VBench2.0 comparisons.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Base model</td><td>Wan2.1-T2V-1.3B-Diffusers</td></tr><tr><td>Scheduler</td><td>UniPC</td></tr><tr><td>Steps</td><td>50</td></tr><tr><td>Guidance scale</td><td>5.0</td></tr><tr><td>Sample/flow shift</td><td>5.0</td></tr><tr><td>Resolution</td><td>832 × 480</td></tr><tr><td>Frames / FPS</td><td>81 / 16</td></tr><tr><td>Seed</td><td>42, with per-video seed 42 + index</td></tr><tr><td>Precision</td><td>float16</td></tr></table>

Table 4: LoRA post-training parameters.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Trainable modules</td><td> ${ \mathfrak { q } } ,$  k,  $\mathbf { V } ,$  output projection LoRA</td></tr><tr><td>Objective</td><td>reward-weighted flow matching MSE</td></tr><tr><td>Reward normalization</td><td>sample weight / global mean weight</td></tr><tr><td>Text length</td><td>512 tokens</td></tr><tr><td>Probe samples</td><td>100 per dimension for the full gradient-probe run</td></tr><tr><td>Probe stages</td><td>0/25/50/75/100%</td></tr><tr><td>Group count</td><td>5</td></tr><tr><td>Group experts</td><td>rank 8, one epoch over the group&#x27;s samples</td></tr><tr><td>Merged adapter</td><td>group-size-weighted dense merge, rank-40 SVD</td></tr></table>

Table 5: On-policy distillation (OPD) settings on Wan2.1.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Student initialization</td><td>merged rank-40 adapter  $\phi _ { 0 }$  (student rank 40)</td></tr><tr><td>Anchor</td><td>frozen merged adapter  $\phi _ { 0 }$ </td></tr><tr><td>Teachers</td><td>five frozen group experts, routed by dimension</td></tr><tr><td>Prompts</td><td>training prompts of the reward pool (videos, weights unused)</td></tr><tr><td>Loss weights</td><td>λOPD = 1.0, λ = 0.1 (Eq. 6)</td></tr><tr><td>Student rollout</td><td>8 sampler steps from fresh noise, no gradient</td></tr><tr><td>Distilled states</td><td>2 interior rollout states per prompt</td></tr><tr><td>Guidance during OPD</td><td>1.0 (no classifier-free guidance)</td></tr><tr><td>Resolution / frames</td><td>832 × 480 / 81</td></tr><tr><td>Optimizer</td><td>AdamW, learning rate  $1 \times 1 0 ^ { - 6 }$  , one pass over the prompts</td></tr><tr><td>Batch</td><td>1 prompt per GPU on 16 GPUs</td></tr><tr><td>Precision</td><td>float16, flow shift 5.0, text length 512</td></tr></table>

Table 6: Compute. GPU-hours are measured on the listed hardware.
<table><tr><td>Stage</td><td>Hardware</td><td>GPU-hours</td><td>Note</td></tr><tr><td>Joint training and gradient probes</td><td>NVIDIA A800</td><td>13.4</td><td></td></tr><tr><td>Five group experts</td><td>NVIDIA A800</td><td>33.5</td><td></td></tr><tr><td>OPD consolidation</td><td>NVIDIA RTX 4090</td><td>≈205</td><td>16 GPUs, 12 h 48 min</td></tr></table>

## B QWEN-BASED REWARD CONSTRUCTION

The reward pipeline converts each generated training video into a scalar sample weight. It uses Qwen3-VL-8B-Instruct in two stages. First, for each prompt and VBench2.0 dimension, Qwen generates a small set of dimension-specific yes/no QA items from a prompt template. Second, Qwen watches the generated video and answers each QA item with either “Yes” or “No”. A sample receives credit for a QA item when the video answer matches the expected answer generated from the text prompt.

For Motion Order Understanding, the QA-generation template instructs Qwen to identify the main subject’s core action events and produce two to four yes/no questions about their temporal order. For example, if the prompt describes a subject clapping, bending down, and then picking up a bag, the generated QA items should ask whether the earlier action happens before the later action, or whether the later action starts after the earlier action finishes. The template explicitly excludes camera movement, shot changes, clothing, background, and other non-action properties so that the reward targets event order rather than general visual quality.

Let a video sample produce N valid QA items. For item j, let $a _ { j } \in \{ 0 , 1 \}$ indicate whether Qwen’s video answer matches the expected answer. The video-level reward is defined as

$$
r = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } a _ { j } .\tag{7}
$$

The reward is then converted into the training weight used in Eq. 2 as

$$
w = \mathrm { c l i p } ( 1 + \alpha ( r - \bar { r } ) , w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } ) ,\tag{8}
$$

where r¯ is the mean reward over retained samples in the same run. The implementation uses $\alpha = 0 . 5$ $w _ { \mathrm { m i n } } = 0 . 5 ,$ and $w _ { \mathrm { m a x } } = 2 . 0 $ . Samples with no valid QA items are skipped; samples with fewer than the recommended number of QA items are retained with a warning if at least one valid QA item remains.

Reward audit. Table 7 summarizes the Wan2.1 reward pool. Stored weights match Eq. 8, and all 17 dimensions are covered. Camera motion has the largest loss from low-QA filtering (512 of 560 retained).

Table 7: Audit of the Wan2.1 reward pipeline.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Produced reward records</td><td>9,512</td></tr><tr><td>Records retained for training</td><td>9,448 (99.33%)</td></tr><tr><td>Reward-pipeline failures</td><td>0</td></tr><tr><td>Duplicate video paths QA-structure anomalies</td><td>0</td></tr><tr><td></td><td>0</td></tr><tr><td>Samples at a weight-clipping bound</td><td>0</td></tr></table>

## C PARTITION CONTROLS

The random partition controls use the same number of groups as $\mathbf { G } ^ { 3 } \mathbf { - } \mathbf { L o R A }$ but are not selected by the final residual-gradient compatibility criterion. They use the same grouped-LoRA training and merge pipeline as $\mathbf { G } ^ { 3 } \mathbf { - } \mathbf { L o R A }$ and are intended to check whether the final result follows from the specific residual-gradient partition rather than from the existence of five smaller adapters alone.

Random partition A uses groups {complex landscape, complex plot, composition, human identity, mechanics}, {dynamic attribute, material, thermotics}, {camera motion, human anatomy, multi-view consistency}, {motion order, motion rationality, human clothes}, and {dynamic spatial relationship, human interaction, instance preservation}.

Random partition B uses groups {dynamic spatial relationship, human anatomy, human clothes, human identity, human interaction, instance preservation, motion order, motion rationality}, {dynamic attribute, material, mechanics, thermotics}, {camera motion, multi-view consistency}, {complex landscape, composition}, and {complex plot}.

The semantic partition uses the official VBench2.0 categories without the diversity dimension: Human Fidelity {human anatomy, human identity, human clothes}; Creativity {composition}; Controllability {dynamic spatial relationship, dynamic attribute, motion order, human interaction, complex landscape, complex plot, camera motion}; Physics {mechanics, thermotics, material, multi-view consistency}; and Commonsense {motion rationality, instance preservation}.

The raw-gradient partition applies average-linkage clustering with $M = 5$ to the raw categorygradient cosine matrix, without removing the global direction. Directly subtracting the global gradient instead of projecting it yields exactly the $\mathrm { G ^ { 3 } \mathrm { - 1 } }$ LoRA partition.

## D DIMENSION-LEVEL VBENCH2.0 TABLE

Table 8: Dimension-level scores under the quick VBench2.0 evaluation protocol. Scores are percentages; ties are counted as neither improved nor degraded in aggregate tables. Grouped r8 is included as an appendix-only merge-rank diagnostic; the main comparison uses Grouped r40 (Ours). Group-only scores each dimension with the expert of its own group (non-deployable diagnostic). +OPD is Grouped r40 after on-policy distillation.
<table><tr><td>Dimension</td><td>Base</td><td>Joint LoRA</td><td>Random A</td><td>Random B</td><td>Group-only</td><td>Grouped r8</td><td>Grouped r40 (Ours)</td><td>+OPD (Ours)</td></tr><tr><td>Human Anatomy</td><td>88.21</td><td>86.27</td><td>87.38</td><td>86.66</td><td>83.88</td><td>85.00</td><td>85.99</td><td>85.94</td></tr><tr><td>Human Clothes</td><td>97.92</td><td>97.96</td><td>95.92</td><td>97.96</td><td>92.00</td><td>83.33</td><td>95.92</td><td>98.40</td></tr><tr><td>Human Identity</td><td>73.37</td><td>71.69</td><td>73.40</td><td>73.01</td><td>68.54</td><td>77.15</td><td>71.59</td><td>71.49</td></tr><tr><td>Composition</td><td>48.10</td><td>46.30</td><td>49.40</td><td>50.20</td><td>53.60</td><td>46.30</td><td>47.80</td><td>47.30</td></tr><tr><td>Mechanics</td><td>56.52</td><td>61.70</td><td>54.35</td><td>61.70</td><td>72.92</td><td>70.00</td><td>57.78</td><td>58.49</td></tr><tr><td>Material</td><td>36.67</td><td>29.03</td><td>34.48</td><td>31.03</td><td>30.00</td><td>45.83</td><td>32.14</td><td>32.70</td></tr><tr><td>Thermotics</td><td>47.83</td><td>51.16</td><td>48.89</td><td>48.94</td><td>55.32</td><td>66.67</td><td>51.11</td><td>51.69</td></tr><tr><td>Multi-view Consistency</td><td>42.96</td><td>47.60</td><td>55.37</td><td>43.91</td><td>73.56</td><td>30.54</td><td>51.95</td><td>60.02</td></tr><tr><td>Dynamic Spatial Relationship</td><td>34.00</td><td>38.00</td><td>40.00</td><td>34.00</td><td>32.00</td><td>30.00</td><td>38.00</td><td>38.00</td></tr><tr><td>Dynamic Attribute</td><td>20.00</td><td>14.00</td><td>16.00</td><td>16.00</td><td>12.00</td><td>16.00</td><td>20.00</td><td>24.00</td></tr><tr><td>Motion Order Understanding</td><td>24.00</td><td>26.00</td><td>24.00</td><td>26.00</td><td>22.45</td><td>28.57</td><td>32.00</td><td>34.04</td></tr><tr><td>Human Interaction</td><td>66.00</td><td>58.00</td><td>58.00</td><td>50.00</td><td>58.00</td><td>60.03</td><td>62.00</td><td>68.00</td></tr><tr><td>Complex Landscape</td><td>20.00</td><td>21.60</td><td>20.80</td><td>21.20</td><td>22.80</td><td>18.80</td><td>22.40</td><td>22.80</td></tr><tr><td>Complex Plot</td><td>10.40</td><td>13.60</td><td>8.40</td><td>16.40</td><td>10.00</td><td>9.20</td><td>12.40</td><td>17.47</td></tr><tr><td>Camera Motion</td><td>12.00</td><td>20.00</td><td>8.00</td><td>10.00</td><td>6.00</td><td>16.00</td><td>12.00</td><td>12.00</td></tr><tr><td>Motion Rationality Instance Preservation</td><td>38.00</td><td>36.00 80.00</td><td>38.00 76.00</td><td>36.00 76.00</td><td>42.00 82.00</td><td>38.00</td><td>40.00</td><td>40.20</td></tr><tr><td></td><td>84.21</td><td></td><td></td><td></td><td></td><td>84.51</td><td>86.00</td><td>86.00</td></tr><tr><td>Mean</td><td>47.07</td><td>46.99</td><td>46.38</td><td>45.82</td><td>48.06</td><td>47.41</td><td>48.18</td><td>49.91</td></tr></table>

The rank-8 merge is lower than the rank-40 merge on the average score (47.41 vs. 48.18) and is included only to diagnose merge-rank sensitivity. Its dimension-level pattern is uneven: it improves mechanics, material, and thermotics, but drops multi-view consistency and human clothes consistency sharply. This supports using Grouped r40 (Ours) as the main grouped adapter in Table 1.

## E GRADIENT COMPATIBILITY DIAGNOSTICS

Figure 5 shows the average-linkage clustering tree used to obtain the final grouped LoRA data split. The distance is $1 - \bar { S } .$ , where S<sup>¯</sup> is the mean projection-residual gradient similarity over five probe stages. This figure complements the heatmap in the main text by showing the hierarchical structure before cutting the tree into five groups.

Figure 6 shows the stage-wise projection-residual similarity matrices at 0%, 25%, 50%, 75%, and 100% of the reference joint training trajectory. All five heatmaps use the final $M = 5$ grouping order.

![](images/2af50c113959a52805ea7d76c34eee194541182db2bb199993dff04ed938c633.jpg)  
Figure 5: Hierarchical clustering over mean residual-gradient compatibility. The $M = 5$ cut separates five groups with positive within-group residual similarity and avoids negative within-group pairs.

Their role is diagnostic: they indicate whether the compatibility structure used for grouping is a persistent training signal rather than a one-off probe artifact.

Shared-direction removal. Table 9 compares the raw gradient, direct subtraction of the global gradient, and the projection residual of Eq. 4. Raw gradients make every pair positive. Direct subtraction and projection give nearly identical matrices (Pearson/Spearman 0.998/0.997) and the same $M = 5$ partition (adjusted Rand index 1.0), so they define the same downstream model.

Table 9: Grouping signals on Wan2.1. Separation is the within-group minus between-group mean similarity of the resulting $M = 5$ partition, computed on the corresponding matrix.
<table><tr><td>Signal</td><td>Negative pairs</td><td>Separation</td><td>VBench2.0 mean</td></tr><tr><td>Raw gradient</td><td>0/136</td><td>0.046</td><td>47.35</td></tr><tr><td>Direct subtraction</td><td>76/136</td><td>0.635</td><td>48.18 (same partition)</td></tr><tr><td>Projection residual  $( \mathrm { E q . 4 } )$ </td><td>75/136</td><td>0.633</td><td>48.18</td></tr></table>

Choice of the number of groups. Table 10 compares $M = 4$ and $M = 5$ on the mean residual matrix. The choice uses training gradients only, not evaluation scores.

Table 10: Clustering diagnostics for the number of groups on Wan2.1.
<table><tr><td></td><td>Groups Within-group mean</td><td></td><td>Separation Negative within-group pairs</td></tr><tr><td> $M = 4$ </td><td>0.345</td><td>0.559</td><td>3</td></tr><tr><td> $M = 5$ </td><td>0.473</td><td>0.633</td><td>0</td></tr></table>

Prevalence across backbones and probe cohorts. Table 11 reports residual conflict for the Wan2.1 probes and for three non-overlapping CogVideoX-2B probe cohorts. On Wan2.1, 65 of 136 pairs are negative at all five training stages and 71 at four or more, and the stage-wise matrices have mean Pearson/Spearman correlation 0.966/0.962.

![](images/f2e27b617ecd91b360f984f132350a01469b4a1fbc7edad8ffa41d3ef36cc3a3.jpg)

![](images/0a49a0df7abbb072dd66df041f83b231fd79301b5b9bb11d0991a03916ce82b2.jpg)

![](images/1eb67f8a00c1abd31f44ac8a9908c3de6561eb9dd143d231cf6d36e9e1520aab.jpg)

![](images/86dcfe0e8f7ad41338747f240950487a22a35a0b100b8e218bd9f63769bec9d7.jpg)

![](images/77ba248046b5db19ce22e28bb8112223ba82ef8971d32c4a5c1dca4614ababff.jpg)  
Figure 6: Stage-wise residual-gradient compatibility. Each heatmap is computed after projecting out the global mixed-gradient direction at the corresponding probe stage. The repeated block structure provides evidence that the grouping is not determined by a single noisy probe.

Table 11: Residual-gradient conflict across backbones and probe cohorts.
<table><tr><td>Backbone / probe cohort</td><td>Negative pairs</td><td>Within-group</td><td>Between-group</td><td>Separation</td></tr><tr><td>Wan2.1 probes</td><td>75/136 (55.1%)</td><td>+0.473</td><td>-0.160</td><td>+0.633</td></tr><tr><td>CogVideoX-2B main probes</td><td>84/136 (61.8%)</td><td>+0.154</td><td>-0.102</td><td>+0.257</td></tr><tr><td>CogVideoX-2B cohort A</td><td>81/136 (59.6%)</td><td>+0.177</td><td>-0.121</td><td>+0.298</td></tr><tr><td>CogVideoX-2B cohort B</td><td>87/136 (64.0%)</td><td>+0.191</td><td>-0.121</td><td>+0.312</td></tr></table>

Probe resampling. Three probe manifests, each independently sampling 100 examples per dimension with no overlap with held-out examples, were compared with the full reference matrix at the final Wan2.1 checkpoint (Table 12). Strong relations are the most reproducible.

Table 12: Stability of the projection-residual matrix under probe resampling on Wan2.1.
<table><tr><td>Comparison</td><td>Pearson</td><td>Spearman</td><td>Sign agr. (all)</td><td>Sign agr. (strong)</td></tr><tr><td>Resamples vs. full reference</td><td>0.847</td><td>0.834</td><td>0.816</td><td>0.853</td></tr><tr><td>Across the three resamples</td><td>0.728</td><td>0.703</td><td>0.755</td><td>0.787</td></tr></table>

Held-out transfer. Gradients were estimated from 100 probe samples per dimension; crossdimension loss changes after a small step along each category gradient were measured on a disjoint held-out set. Table 13 reports the rank correlation between residual similarity and held-out transfer. The transfer rankings are nearly identical across step sizes (pairwise Spearman 0.9992–0.9997).

Table 13: Residual similarity predicts held-out cross-dimension transfer on Wan2.1.
<table><tr><td>Relative step size</td><td>Spearman  $\rho$ </td><td>Permutation  $p$ </td></tr><tr><td>0.0005</td><td>0.256</td><td>0.0028</td></tr><tr><td>0.001</td><td>0.255</td><td>0.0038</td></tr><tr><td>0.002</td><td>0.252</td><td>0.0028</td></tr></table>

## F DENSE DELTA-SVD LORA MERGE

The grouped rank-40 adapter is produced by merging the five group-specific LoRA experts in dense weight-delta space and then recompressing the result back into LoRA factors. For a linear layer with frozen base weight W and a LoRA expert m, let $A _ { m } \in \mathbb { R } ^ { r _ { m } \times d _ { \mathrm { i r } } }$ <sup>n</sup> and $B _ { m } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r _ { m } }$ denote the learned LoRA factors. The dense update represented by the expert is defined as

$$
\Delta W _ { m } = B _ { m } A _ { m } ,\tag{9}
$$

where the implementation uses unit LoRA scaling for the merge. The group experts are then averaged in dense delta space as

$$
\Delta { W } _ { \mathrm { m e r g e } } = \sum _ { m = 1 } ^ { M } \alpha _ { m } \Delta { W } _ { m } , \qquad \sum _ { m = 1 } ^ { M } \alpha _ { m } = 1 .\tag{10}
$$

For the rank-40 grouped adapter, the weights are proportional to the number of VBench2.0 dimensions assigned to each group. With group sizes [4, 4, 3, 4, 2], the corresponding weights are [4/17, 4/17, 3/17, 4/17, 2/17].

The dense merged update is converted back into a rank-r LoRA module through truncated singular value decomposition. For each merged layer, this is computed as

$$
\Delta W _ { \mathrm { m e r g e } } \approx U _ { r } \Sigma _ { r } V _ { r } ^ { \top } ,\tag{11}
$$

with requested rank r = 40. The recompressed LoRA factors are defined as

$$
B _ { \mathrm { n e w } } = U _ { r } \Sigma _ { r } ^ { 1 / 2 } , \qquad A _ { \mathrm { n e w } } = \Sigma _ { r } ^ { 1 / 2 } V _ { r } ^ { \top } .\tag{12}
$$

This gives $B _ { \mathrm { n e w } } A _ { \mathrm { n e w } } \approx \Delta W _ { \mathrm { m e r g e } }$ while preserving the standard LoRA loading interface. Because the five rank-8 experts span at most 40 dimensions per layer, the rank-40 recompression reproduces $\Delta W _ { \mathrm { m e r g e } }$ up to numerical error. On CogVideoX-2B, five rank-128 experts are merged exactly into a rank-640 adapter.

## G MERGE RETENTION EVALUATION

To separate group training quality from merge-induced compression, we run a retention evaluation on the five group-specific LoRA experts. Each group-only LoRA is evaluated only on the VBench2.0 dimensions assigned to that group, and the resulting dimension scores are compared with the base model and the merged rank-40 adapter under a matched quick-evaluation protocol. This is an oraclestyle diagnostic rather than a deployable single-model result: it asks whether the merged adapter preserves the capabilities learned by the group experts, rather than only whether the final merged model improves the average score.

Table 14: Group-level retention summary. Group-only LoRAs are an oracle-style diagnostic evaluated only on assigned dimensions; Merged r40 (Ours) preserves the average while still compressing some group-specific skills. Bold indicates the highest score in each row.
<table><tr><td>Scope</td><td>Base</td><td>Merged r40 (Ours)</td><td>Group-only</td><td>Merged – Group</td><td>+OPD (Ours)</td></tr><tr><td>All 17 dims</td><td>47.07</td><td>48.18</td><td>48.06</td><td>+0.12</td><td>49.91</td></tr><tr><td>Group 1</td><td>60.03</td><td>62.98</td><td>57.11</td><td>+5.87</td><td>64.11</td></tr><tr><td>Group 2</td><td>66.39</td><td>64.89</td><td>63.10</td><td>+1.79</td><td>66.41</td></tr><tr><td>Group 3</td><td>26.70</td><td>27.40</td><td>27.47</td><td>-0.07</td><td>27.37</td></tr><tr><td>Group 4</td><td>36.86</td><td>38.80</td><td>42.72</td><td>-3.92</td><td>42.10</td></tr><tr><td>Group 5</td><td>33.46</td><td>35.09</td><td>41.46</td><td>-6.37</td><td>37.98</td></tr></table>

Table 14 shows that the rank-40 merge preserves most average group-only performance but compresses some specialized skills. Groups 1 and 2 are better after merging than in their group-only form, suggesting that some dimensions benefit from updates learned by other groups. In contrast, Groups 4 and 5 lose 3.92 and 6.37 points after merging. The largest dimension-level compression occurs on multi-view consistency and mechanics, as shown in Table 15.

Table 15: Selected dimension-level retention results from the group-only LoRA evaluation. Scores are percentages under the quick VBench2.0 protocol.
<table><tr><td>Group</td><td>Dimension</td><td>Base</td><td>Merged r40 (Ours)</td><td>Group-only</td><td>Merged – Group</td><td>+OPD (Ours)</td></tr><tr><td>Group 1</td><td>Dynamic Spatial Relationship</td><td>34.00</td><td>38.00</td><td>32.00</td><td>+6.00</td><td>38.00</td></tr><tr><td>Group 1</td><td>Human Clothes</td><td>97.92</td><td>95.92</td><td>92.00</td><td>+3.92</td><td>98.40</td></tr><tr><td>Group 1</td><td>Instance Preservation</td><td>84.21</td><td>86.00</td><td>82.00</td><td>+4.00</td><td>86.00</td></tr><tr><td>Group 1</td><td>Motion Order Understanding</td><td>24.00</td><td>32.00</td><td>22.45</td><td>+9.55</td><td>34.04</td></tr><tr><td>Group 2</td><td>Human Anatomy</td><td>88.21</td><td>85.99</td><td>83.88</td><td>+2.11</td><td>85.94</td></tr><tr><td>Group 2</td><td>Human Identity</td><td>73.37</td><td>71.59</td><td>68.54</td><td>+3.05</td><td>71.49</td></tr><tr><td>Group 2</td><td>Human Interaction</td><td>66.00</td><td>62.00</td><td>58.00</td><td>+4.00</td><td>68.00</td></tr><tr><td>Group 2</td><td>Motion Rationality</td><td>38.00</td><td>40.00</td><td>42.00</td><td>-2.00</td><td>40.20</td></tr><tr><td>Group 3</td><td>Camera Motion</td><td>12.00</td><td>12.00</td><td>6.00</td><td>+6.00</td><td>12.00</td></tr><tr><td>Group 3</td><td>Complex Landscape</td><td>20.00</td><td>22.40</td><td>22.80</td><td>-0.40</td><td>22.80</td></tr><tr><td>Group 3</td><td>Composition</td><td>48.10</td><td>47.80</td><td>53.60</td><td>-5.80</td><td>47.30</td></tr><tr><td>Group 4</td><td>Dynamic Attribute</td><td>20.00</td><td>20.00</td><td>12.00</td><td>+8.00</td><td>24.00</td></tr><tr><td>Group 4</td><td>Material</td><td>36.67</td><td>32.14</td><td>30.00</td><td>+2.14</td><td>32.70</td></tr><tr><td>Group 4</td><td>Multi-view Consistency</td><td>42.96</td><td>51.95</td><td>73.56</td><td>-21.60</td><td>60.02</td></tr><tr><td>Group 4</td><td>Thermotics</td><td>47.83</td><td>51.11</td><td>55.32</td><td>-4.21</td><td>51.69</td></tr><tr><td>Group 5</td><td>Complex Plot</td><td>10.40</td><td>12.40</td><td>10.00</td><td>+2.40</td><td>17.47</td></tr><tr><td>Group 5</td><td>Mechanics</td><td>56.52</td><td>57.78</td><td>72.92</td><td>-15.14</td><td>58.49</td></tr><tr><td>All</td><td>Mean</td><td>47.07</td><td>48.18</td><td>48.06</td><td>+0.12</td><td>49.91</td></tr></table>

Selecting post hoc, for each dimension, the better of the assigned expert and the merged adapter yields a diagnostic mean of 51.07. Table 16 rules out rank truncation and broad cancellation of expert deltas as the main causes of specialist compression. Group 5 (complex plot and mechanics) has a small learned delta and a 2/17 merge weight, consistent with dilution of the mechanics specialist.

Multi-view consistency belongs to Group 4, whose norm share is not small, so its compression is more consistent with averaging or activation-level interference.

Table 16: Parameter-space merge diagnostics.
<table><tr><td>Diagnostic</td><td>Value</td></tr><tr><td>Rank-40 SVD reconstruction error (Wan2.1)</td><td> $1 . 4 6 4 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Pairwise cosine between global expert deltas</td><td>0.730–0.938</td></tr><tr><td>Modules with a negative expert-pair cosine</td><td>0/240</td></tr><tr><td>Group-size-weighted norm retention</td><td>0.946</td></tr><tr><td>Group 5 share of the weighted expert-delta norm</td><td>2.08%</td></tr><tr><td>Cog VideoX-2B rank-640 merge error (dense delta / output)</td><td> $3 . 4 4 \times 1 0 ^ { - 7 } / 7 . 0 5 \times 1 0 ^ { - 7 }$ </td></tr></table>

## H ON-POLICY DISTILLATION DETAILS

Each OPD step processes one training prompt per GPU. (i) The current student samples an 8-step trajectory from fresh Gaussian noise without gradient, and two interior latent states $z _ { \tau }$ are stored with their timesteps. (ii) The prompt’s dimension selects its group expert $\psi _ { m ( p ) } ;$ the frozen expert and the frozen merged anchor ϕ predict velocities at the stored states. (iii) The student predicts velocities at the same states, and Eq. 6 is minimized with respect to the student LoRA only. All LoRA states share the same frozen base model and are swapped in place, so no additional backbone copy is required. The loss does not use the training videos, the flow-matching target of Eq. 2, or the reward weights. Table 5 lists the hyperparameters, and Table 17 reports the change of every dimension relative to the merged adapter. The paired 95% bootstrap interval of the mean change is [+0.42, +3.05].

Table 17: Change after OPD relative to the merged rank-40 adapter on Wan2.1.
<table><tr><td>Dimension</td><td> $\Delta$ </td><td>Dimension</td><td>∆</td></tr><tr><td>Multi-view Consistency</td><td>+8.07</td><td>Thermotics</td><td>+0.58</td></tr><tr><td>Human Interaction</td><td>+6.00</td><td>Material</td><td>+0.56</td></tr><tr><td>Complex Plot</td><td>+5.07</td><td>Complex Landscape</td><td>+0.40</td></tr><tr><td>Dynamic Attribute</td><td>+4.00</td><td>Motion Rationality</td><td>+0.20</td></tr><tr><td>Human Clothes</td><td>+2.48</td><td>Dynamic Spatial Relationship</td><td>0.00</td></tr><tr><td>Motion Order Understanding</td><td>+2.04</td><td>Camera Motion</td><td>0.00</td></tr><tr><td>Mechanics</td><td>+0.71</td><td>Instance Preservation</td><td>0.00</td></tr><tr><td>Human Anatomy</td><td>-0.05</td><td>Human Identity</td><td>-0.10</td></tr><tr><td>Composition</td><td>-0.50</td><td>Mean</td><td>+1.73</td></tr></table>

Relative to the six specialist gaps of the merged adapter, OPD recovers 8.07 of 21.61 points on multi-view consistency, 0.71 of 15.14 on mechanics, 0.58 of 4.21 on thermotics, 0.20 of 2.00 on motion rationality, and all 0.40 on complex landscape, while composition moves 0.50 further from its expert.

## I COGVIDEOX-2B EXPERIMENT

The CogVideoX-2B experiment reruns the full procedure on a second backbone with its native diffusion-denoising objective: self-generated training videos, QA rewards, gradient probes, grouping, expert training, and merging. Joint and grouped training use the same 9,520 videos, prompts, reward weights, and LoRA and optimizer configuration (LoRA rank 128, α = 128, learning rate $\mathrm { \bar { 1 } \times 1 0 ^ { - 4 } ) }$ each sample is seen exactly once, because the five experts train on disjoint subsets whose sizes sum to 9,520. The five rank-128 experts are merged exactly into a rank-640 adapter $( \alpha = 6 4 0 )$ . Scores are VBench2.0 macro means over 17 dimensions; multi-view consistency is 0 for all CogVideoX methods.

Joint LoRA is already below the base model at 25% exposure, and even its best retained checkpoint remains below both the base model and the grouped adapter, so early stopping would not change the comparison. The degradation is dimension-selective: human anatomy (48.5 to 98.4), human identity (70.9 to 97.9), and thermotics (46.8 to 56.0) improve under Joint LoRA, while composition, camera motion, mechanics, human interaction, motion order, and motion rationality deteriorate. The grouped–Joint difference of +13.04 points has a paired prompt-cluster 95% bootstrap interval of $[ + 9 . 0 8 , + 1 6 . 9 2 ]$

Table 18: CogVideoX-2B Joint LoRA across retained checkpoints (VBench2.0 macro score, ×100).
<table><tr><td>Model / checkpoint</td><td>Training exposure</td><td>Score</td></tr><tr><td>Base</td><td>0%</td><td>39.87</td></tr><tr><td>Joint LoRA</td><td>25%</td><td>36.78</td></tr><tr><td>Joint LoRA</td><td>50%</td><td>31.58</td></tr><tr><td>Joint LoRA</td><td>75%</td><td>32.12</td></tr><tr><td>Joint LoRA</td><td>100%</td><td>27.95</td></tr><tr><td>Grouped LoRA (Ours)</td><td>100%</td><td>40.99</td></tr></table>

## J INDEPENDENT EVALUATION WITH VIDEOSCORE

Table 19: VideoScore-v1.1 on the matched Wan2.1 evaluation videos. The derived mean averages the five aspects; its paired 95% CI is [0.140, 0.192] with $P ( \Delta > 0 ) = 1 . 0 0 0$
<table><tr><td>Aspect</td><td>Base</td><td>Grouped LoRA (Ours)</td><td> $\Delta$ </td></tr><tr><td>Visual quality</td><td>3.029</td><td>3.196</td><td>+0.168</td></tr><tr><td>Temporal consistency</td><td>2.748</td><td>2.931</td><td>+0.183</td></tr><tr><td>Dynamic degree</td><td>3.225</td><td>3.380</td><td>+0.155</td></tr><tr><td>Text alignment</td><td>2.911</td><td>3.041</td><td>+0.130</td></tr><tr><td>Factual consistency</td><td>2.659</td><td>2.851</td><td>+0.192</td></tr><tr><td>Five-aspect mean</td><td>2.914</td><td>3.080</td><td>+0.166</td></tr></table>

Bootstrap protocol. For every paired comparison, prompts are resampled with replacement as clusters, both videos of a prompt stay together, and the same prompt indices are used for both methods. We report the point estimate, the 95% percentile interval over 10,000 resamples, and the fraction of resamples with a positive difference. For Grouped LoRA (Ours) versus Base on VBench2.0 this gives +1.11 points (bootstrap mean +1.09), a 95% interval of [−0.08, +2.26], and $P ( \Delta > 0 ) = 0 . 9 6 6$