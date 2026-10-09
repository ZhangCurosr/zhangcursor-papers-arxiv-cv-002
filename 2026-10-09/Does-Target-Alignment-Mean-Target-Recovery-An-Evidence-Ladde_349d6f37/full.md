# Does Target Alignment Mean Target Recovery? An Evidence-Ladder Study of Adversarial Claims on Contrastive Encoders

Tao Yang Singapore University of Technology and Design tao yang@mymail.sutd.edu.sg

Jianying Zhou Singapore University of Technology and Design jianying zhou@sutd.edu.sg

October 2026

## Abstract

Adversarial attacks on vision–language models optimize an image toward a text target, then cite the attacked model’s similarity score as evidence of success. We ask whether that score—victim-space target alignment (VTS)—predicts recovery of the target by an independent model. We first build a measurement instrument: supervised judges outside the attacked geometry, real-target blend controls, shuffled-target negatives, and a reference level derived from a 50% target-image blend. Two preregistered studies then compare six contrastive encoders under a matched attack at three perturbation budgets. Robustly trained encoders (FARE, TeCoA, PMG, TRADES) transfer substantially more independent evidence than vanilla CLIP or SigLIP; all eight contrasts reject at the bootstrap floor. However, no cell reaches the blend-derived reference level. The three best cells fall within its replication band, leaving practical recovery undecided. Within robust encoders, per-sample alignment gain correlates with evidence gain $( \rho = 0 . 2 4 \ – 0 . 5 1 )$ ; within vanilla CLIP the correlation is consistent with zero. Across encoders we find no monotone alignment–evidence relation. VTS is therefore informative only within a fixed robust encoder, and we provide a reporting protocol in its place.

## 1 Introduction

Optimizing an image to increase its similarity to a text target in a contrastive model’s embedding space is a standard step in adversarial attacks on vision–language systems [11, 16, 41]. After optimization, the attacker reports the resulting score as evidence that the attack succeeded in inducing the target concept. The practice is widespread because the quantity is cheap to compute, directly optimized, and available to any white-box attack. But it is computed by the same encoder being optimized, so it rises under attack regardless of whether any other model can recognize the intended target. This circularity is not a hypothetical concern: prior work has shown that adversarial images can score high on victim-side alignment while failing entirely under independent evaluation. Zhang et al. [42] found that adversarial images yield 0% target recovery under an independent classifier while images generated from their embeddings yield 64%, separating input-level alignment from generative recovery.

Despite this known gap, victim-side alignment remains the default evidence in adversarial semantic claims on contrastive models. Attacks are compared by their alignment scores, defenses are evaluated by how much they reduce alignment, and “semantic” conclusions are drawn from a quantity that the attacker controls.

What remains unresolved is a narrower and more actionable question: when does increased alignment in the attacked model predict independently measured recovery?

We approach this as a measurement problem rather than an attack-design problem. Our instrument pairs supervised judges, architecturally outside the contrastive family, with graded real-target blend controls and shuffled-target negatives, anchored to a reference level set by the mean independent evidence of a 50% target-image blend. Two preregistered studies evaluate vanilla CLIP [26], four adversarially fine-tuned variants (FARE, TeCoA, PMG, TRADES), and SigLIP [38] (a second contrastive family with a sigmoid objective), under a matched fixed-target attack at ϵ ∈ {4, 16, 64}/255, on held-out images with regenerated target mappings. A preregistered recipe comparison additionally tests whether stronger standard attacks (targeted PGD, C&W) change any conclusion.

Three results emerge. First, robustly fine-tuned encoders transfer more independent target evidence than non-adversarially trained ones. All eight robust-versus-naive contrasts reject at the bootstrap floor. SigLIP, which differs from vanilla CLIP in objective, scale, and training recipe, stays at vanilla-like levels. Second, no cell’s mean evidence reaches the blend reference level. The three nearest cells fall within the replication band of that level, so the question of practical recovery remains open rather than settled. Third, alignment gain predicts evidence gain within robust encoders $\left( \rho = 0 . 2 4 \mathrm { - } 0 . 5 1 \right) _ { \it { \Omega } }$ ) but not within vanilla CLIP (consistent with zero under a post-hoc approximation), and we find no monotone alignment–evidence relation across encoders. The naive-versus-robust inversion is real but does not extend to SigLIP.

An evidence taxonomy (L0 target intent through L4 human grounding) frames what each type of claim requires; our results are scoped to machine-level evidence (L1 and L2). Our contributions are:

1. A calibrated measurement instrument for independent target recovery, combining supervised judges outside the contrastive family, graded real-target blend positive controls, shuffled-target negatives, and a pre-specified reference level derived from a 50% target-image blend. All controls are preregistered and their outputs are independently validated.

2. A cross-encoder boundary map: the first systematic evidence that adversarially fine-tuned encoders transfer more independent target evidence than non-adversarially trained ones, quantified across four robust recipes and two contrastive families (six encoders total) at three perturbation budgets under preregistered contrasts.

3. A two-level diagnosticity finding: victim-space alignment is informative within robust encoders (ρ = 0.24–0.51), consistent with zero within vanilla CLIP, and we find no monotone relation across encoders. This separates the metric’s within-encoder use from its (unsupported) cross-encoder use.

4. A reporting protocol for adversarial semantic claims, specifying the evidence required at each level and the calibration steps that prior transfer-evaluation approaches lack.

## 2 Related Work

Targeted and universal attacks on contrastive models. Universal adversarial triggers originate in text classification [36]; discrete gradient-based suffix search was systematized for LLM jailbreaks by GCG [43]. Recent attacks on CLIP-scale models include targeted generators [41] and universal perturbations [11, 16]. Several evaluate on models beyond the surrogate (AnyAttack transfers to unseen target VLMs; Springer et al. [32] study robust-source targeted transfer), but their endpoint is attack or transfer success, not target-specific evidence calibrated against positive and negative controls

Human evaluation and unrestricted attacks. Bhattad et al. [5] craft unrestricted semantic-manipulation attacks that preserve source-class content. Elsayed et al. [10] show that adversarial perturbations bias timelimited human choices, reporting 57–89% targeted transfer to models outside the optimization. Veerabadran et al. [35] demonstrate that bounded perturbations bias human category judgments toward the chosen class. Zhang et al. [39] generate attacks whose original concept remains identifiable. These works establish that human evaluation of adversarial content has precedent; our contribution is the calibration protocol (graded controls, shuffle correction, a frozen reference level) applied across contrastive encoders and budgets, which none of them provide.

Robust features and gradient alignment. Adversarial examples exploit predictive but non-robust features [18]; robust classifiers carry perceptually aligned gradients [19, 28], which can be promoted without adversarial training while remaining associated with robustness [15]. Off-manifold robustness helps explain this property [33]. Gradient alignment in CLIP-family encoders has been characterized [14]; on-manifold perturbations relate robustness to generalization [21, 34]; the joint space has a modality gap [20, 22]. None of this work conditions independent recoverability on victim robustness; the nearest results show that adversarially trained sources produce more transferable attacks [2] and that robustified CLIP induces better perceptual metrics [8]; neither endpoint is calibrated target evidence.

Evaluation validity. Obfuscated gradients taught the field that evaluation under optimization requires adversarial scrutiny [1]; attack implementations and judge protocols remain sources of distortion [6, 7, 17, 23, 30, 31], and human-grounded frameworks audit automated judges against perception [12]. Psychometric validity theory is being imported into ML evaluation [3, 13, 27], covering both language and vision. No prior work addresses adversarial semantic claims with the calibrated protocol used here. We extend the evaluation-validity line from “is the attack real?” to “does the semantic metric mean what it is used to mean?” Universal-suffix work in the LLM domain independently supports non-semantic readings of trigger success [4].

## 3 An Evidence Taxonomy for Adversarial Claims

Adversarial “semantic” claims conflate several distinct questions. We separate them into levels: L0 target intent (WordNet or other ontology), L1 victim-space alignment (VTS), L2 independent-model recovery (ITE), L3 downstream propagation, and L4 human grounding. Passing one level does not imply passing another. Our results address L1 and L2; we do not claim L3 or L4 evidence.

Table 1: Evidence levels and what each does not establish.
<table><tr><td>Level</td><td>Construct</td><td>Admissible evidence</td><td>Does not establish</td></tr><tr><td>L0</td><td>Target intent</td><td>WordNet/ontology map</td><td>any change in the image</td></tr><tr><td>L1</td><td>Victim alignment</td><td>frozen-target cosine/margin</td><td>independent recognition</td></tr><tr><td>L2</td><td>Independent recovery</td><td>validated external judges</td><td>human perception</td></tr><tr><td>L3</td><td>Downstream propagation</td><td>caption/VQA/retrieval endpoints</td><td>human perception</td></tr><tr><td>L4</td><td>Human grounding</td><td>preregistered human study</td><td>universal meaning</td></tr></table>

Terminology. VTS (victim-space target similarity) is cosine to the frozen target text embedding in the victim. ITE (independent target evidence) is the shuffle-corrected change in the target’s logit margin on an independent judge. ITR (rank recovery, anchor in top-5) is corroborating secondary. Calling an L1 quantity “semantic injection” or “semantic coherence” without L2 or higher evidence is a category error under this framework; we apply the same rule to our own claims.

![](images/37fdf88dc500ef224b3ecbb6fe4a563c19e0b9b5c8812723cdf69ecd56b91a03.jpg)  
Figure 1: Evidence levels. Adjacent levels do not imply one another.

This taxonomy draws on validity-evidence thinking from measurement theory [9, 25], adapted for adversarial evaluation. Concurrent work imports the same theory for ML evaluation more broadly [3, 13, 27], covering both language and vision; ours applies it specifically to adversarial semantic claims on vision– language models, with the calibrated protocol (blend positive controls, shuffled negatives, a frozen reference level) that prior frameworks do not provide. The mapping to Messick’s validity aspects (L1 as structural, L2 as external/convergent, L3 as consequential, L4 as substantive-plus-content) is heuristic rather than established; it serves as a bridge, not a derivation.

## 4 Methods

## 4.1 Study A: instrument validation

Judges. The primary judge is ResNet-50 (ImageNet-1k, supervised), chosen for architectural and training disjointness from the ViT-B/32 victims. The secondary judge, ViT-B/16 (supervised), is reported in the artifacts; it shares the ViT family with the victims, so it corroborates direction and ordering but not scale. Both judges operate on the same 1000-class ImageNet label space, which is the space our evaluation classes map into.

Controls. PC1 (judge competence): clean target-class exemplars establish per-class coverage. PC2 (graded positive control): source images blended with a real target-class image at doses α ∈ {0.25, 0.5, 0.75, 1.0}; the blend constructor is never used as a judge. NC (negatives): clean sources and shuffled targets. A null attack result is interpretable only when these pass on the same endpoint and judge.

Statistic. For judge logits $z _ { c } ,$ , the target logit margin is $m _ { a } = z _ { a } - \mathrm { l o g m e a n e x p } _ { c \neq a } z _ { c }$ . The intended change is $D _ { a } = m _ { a } ( x _ { \mathrm { a d v } } ) - m _ { a } ( x )$ ; the target-specific statistic is $S _ { a } = D _ { a } - \mathbb { E } _ { a ^ { \prime } } [ \dot { m _ { a ^ { \prime } } } ( x _ { \mathrm { a d v } } ) - m _ { a ^ { \prime } } ( x ) ]$ where the expectation runs over 16 shuffled targets $a ^ { \prime }$ per image (drawn without replacement from the 1000 judge labels, excluding a and the true class y). The image is never regenerated under shuffling. We denote $S _ { a }$ the ITE.

Reference level. $S _ { \mathrm { m i n } }$ is anchored at the $\alpha ^ { * } = 0 . 5 0$ blend dose (half of each source pixel replaced by a real target image) and was frozen, with hashes, before Study B (Table 2). A cell is $L 2 - p o s i t i \nu e$ at the anchor iff its mean $\mathrm { I T E } \geq S _ { \mathrm { m i n } }$ and its class-cluster bootstrap 95% CI lower bound exceeds zero. We treat $S _ { \mathrm { m i n } }$ as a pre-specified, reproducible reference level, not a demonstrated utility threshold; utility calibration would require an external task payoff.

## 4.2 Study B: confirmatory core

Victims. Vanilla CLIP, ${ \mathrm { F A R E } } ,$ and TeCoA (all ViT-B/32). TeCoA was trained at $\scriptstyle \epsilon _ { \mathrm { t r a i n } } = 1 / 2 5 5 [ 2 4 ] ;$ our FARE checkpoint is the $\epsilon _ { \mathrm { t r a i n } } { = } 4 / 2 5 5$ release $\mathrm { ( F A R E ^ { 4 } }$ ; hash-pinned in the frozen manifest) [29]. We evaluate at $\epsilon \in \{ 4 , 1 6 , 6 4 \} / 2 5 5 .$ , which spans 1–16× the FARE and 4–64× the TeCoA training budgets.

Attack. The attack minimizes a two-term objective. The first term increases target alignment: − $0 . 4 \langle \hat { f } _ { I } ( x _ { i } + \delta _ { i } ) , \hat { e } _ { a _ { i } } \rangle$ where $\hat { f } _ { I }$ is the ℓ<sub>2</sub>-normalized image embedding and $\hat { e } _ { a _ { i } }$ the frozen target text embedding (both cosines). The second term decreases source-class alignment: $\lambda _ { i } \langle \hat { f } _ { I } ( x _ { i } { + } \delta _ { i } ) , \hat { e } _ { y _ { i } } \rangle$ , where $\hat { e } _ { y _ { i } }$ is the true-class embedding. The weight $\lambda _ { i }$ is 0.5 while the victim’s 100-class head still predicts $y _ { i }$ , otherwise 0.1, re-evaluated after each optimization step. Adam optimizer, learning rate 0.01, 20 steps, $\ell _ { \infty }$ projection each step.

Data. Targets are regenerated per mapping seed {101, 202, 303}. Each seed draws 100 images from the victim-clean-correct intersection of the confirmatory rows [2000, 5000) of a fixed permutation of the ImageNet-100 validation set (exploratory work used rows [0, 2000)). Each cell thus comprises 300 paired image-mapping observations across three seeds; the seeds share the row pool, so unique source images number approximately 285 per cell.

Power and clustering. Cell size was set from pilot estimates of the paired statistic’s standard deviation (1.7–4.1 ITE units on R50). All confirmatory mean contrasts and cell intervals use the class-cluster bootstrap. The within-cell intra-class correlation of ITE over classes is small to moderate (ICC(1,1), unequal-sizecorrected: −0.03 to 0.21). The correlation and equivalence analyses in §5.4 use independent-sample Fisher/TOST approximations, reported as post-hoc sensitivity.

Freeze. Design freeze (grid, $S _ { \mathrm { m i n } }$ , splits, hashes) and code freeze (runner, lockout tests) are separate tagged commits. Outputs are append-only per seed.

## 4.3 Study C: additional encoders

A second, separately frozen design (same rows, seeds, statistic, judge, rule) widens the victim set to six. The robust recipes PMG and TRADES are local adversarially fine-tuned checkpoints, CLIP ViT-B/32-compatible, with recipes partially following Wang et al. [37] and a TRADES-style objective [40]; all checkpoints are hash-pinned in the frozen manifests. SigLIP (siglip-base-patch16-224, sigmoid objective) uses its own text tower and normalization. Both towers use the same prompt in their own tokenizers; VTS is computed within-tower.

Predeclared hypotheses. H C1: PMG and TRADES exceed vanilla at $\epsilon \in \{ 1 6 , 6 4 \}$ (Holm-4, paired). H C2: SigLIP stays vanilla-like (descriptive; no calibrated equivalence margin exists for a new family). H C3: victim-mean VTS anti-ranks ITE across the six encoders (direction-only).

Study C shares its confirmatory rows, mapping seeds, and anchor/shuffle constructions with Study B; the cohorts differ only by the intersection rule. The two Holm families are corrected separately. No pooled cross-study confirmatory analysis is performed; the post-hoc prediction check in $\ S 6$ pools records as a declared exploratory exception. The dataset axis (fine-grained, texture) is deferred: those classes are absent from the ImageNet-1k judge’s label space.

Analysis. Cell-level ITE with class-cluster bootstrap. Four one-sided robust-versus-vanilla contrasts on paired ITE differences under step-down Holm $( \alpha = 0 . 0 5$ , six-contrast family). The family also contains two FARE-versus-TeCoA contrasts; specified one-sided, they were guaranteed-null under the observed ordering, which makes the correction conservative for the four boundary tests. Within-cell Spearman $\rho ( \Delta \mathrm { V T S } , S _ { a } )$ is the preregistered primary form; between-victim means use post-attack VTS. All cells are reported; no re-derivation under alternative rules.

![](images/397f55adff53f383bd0da0cbd2f2197b1de6bc9e00f7ac021b91e9fd34d67c92.jpg)  
Figure 2: Instrument dose–response on the R50 judge: ITE (left axis) and top-1 recovery (right axis) vs. blend dose. The reference level equals the ITE at the $\alpha { = } 0 . 5$ dose.

Table 2: Study A: validation of the L2 independent-evidence instrument (held-out split, $n { = } 2 5 0 )$ . The pre-specified reference level is anchored at the $\alpha ^ { * } { = } 0 . 5 0$ blend dose (half of each source image’s pixels replaced by a real target-class image) and frozen before Study B.
<table><tr><td colspan="2">Quantity Value</td></tr><tr><td>Primary judge</td><td>ResNet-50 (ImageNet-1k, supervised)</td></tr><tr><td>Gate A (PC1/PC2/negative controls)</td><td>PASS</td></tr><tr><td>Held-out ITE at the anchor dose</td><td>5.71 [95% CI 5.12, 6.32]</td></tr><tr><td>Held-out ITR top-5 increase</td><td>0.324 [95% CI 0.268, 0.384]</td></tr><tr><td>Dose-response (Spearman)</td><td> $\rho = 1 . 0 0$  (monotone)</td></tr><tr><td>Frozen margin  $S _ { \mathrm { m i n } }$ </td><td>5.3711</td></tr></table>

## 5 Results

## 5.1 Instrument validation

Study A passed all controls. The primary judge recovers real target evidence monotonically in blend dose $( \rho = 1 . 0 $ , Figure 2). The anchor-dose ITE is 5.71 [95% CI 5.12, 6.32] on held-out data; the reference level is $S _ { \mathrm { m i n } } = 5 . 3 7 1 1$ (Table 2). Negative controls pass on the same endpoint. This establishes that the instrument measures machine-grounded target evidence under the stated protocol; nothing here addresses human-perceived semantics.

## 5.2 Robustness–evidence contrasts

Independent target evidence rises with budget for every encoder (Table 3; per-cell values in Table 5). At $\epsilon { = } 1 6$ and 64 the ordering is $\mathrm { T e C o A } > \mathrm { F A R E } >$ vanilla; at ϵ=4 FARE and TeCoA do not separate. All four preregistered contrasts reject at the bootstrap floor $( p = 1 / B = 0 . 0 0 0 5$ , step-down Holm), with mean differences from $+ 1 . 8 3 \mathrm { t o } + 4 . 4 5$ ITE units over 300 paired images.

No cell reaches the reference level. Table 5 reports per-cell values. $\mathrm { \ A t \ } \epsilon { = } 4$ , all six encoders cluster near zero ITE (0.08–0.39), with overlapping CIs. $\mathrm { A t } \ \epsilon { = } 1 6$ , the robust recipes separate from vanilla by +2.3 to +2.6 ITE units. $\mathrm { A t } \ \epsilon = 6 4$ , the gap widens to +3.4 to +4.5 units for TeCoA, +3.4 for FARE, and +4.2–+4.9 in Study C for PMG and TRADES. The best cell $( \mathrm { T e C o A } , \epsilon { = } 6 4 )$ has mean ITE 5.244 with CI lower bound 4.74: the CI condition passes, the magnitude condition fails by 2.4%. An independent replication of the anchor dose on a different cohort gave 4.65 (approximately 15% calibration sensitivity). Table 4 gives each cell’s verdict under both anchors: exactly three cells (B:TeCoA@64, C:TeCoA@64, C:PMG@64) sit inside the band and flip; all others are negative under both. No cell passes at the Study-A held-out anchor (5.71) either, so the flip set is stable across the full [4.65, 5.71] range of anchor estimates.

Table 3: Study B boundary contrasts (one-sided robust > base on paired ITE differences, n=300 pairs each; class-cluster bootstrap; step-down Holm over the 6-contrast family, $\alpha { = } 0 . 0 5 )$
<table><tr><td>Contrast</td><td>Pairs</td><td>Mean difference</td><td>Holm reject</td></tr><tr><td> $\mathrm { F A R E - v a n i l l a } \ @ \ \epsilon = 1 6$ </td><td>300</td><td>+1.83</td><td>yes</td></tr><tr><td> $\mathrm { F A R E - v a n i l l a ~ } @ ~ \epsilon = 6 4 ~ $ </td><td>300</td><td>+2.60</td><td>yes</td></tr><tr><td> $\mathrm { T e C o A – v a n i l l a \ } @ \ \epsilon { = } 1 6$ </td><td>300</td><td>+2.55</td><td>yes</td></tr><tr><td> $\mathrm { T e C o A – v a n i l l a ~ } @ \ \in 6 4$ </td><td>300</td><td>+4.45</td><td>yes</td></tr><tr><td> $\mathrm { F A R E - T e C o A } \ @ \ \epsilon = 1 6$ </td><td>300</td><td>-0.72</td><td>no</td></tr><tr><td> $\mathrm { F A R E - T e C o A } \ @ \ \epsilon = 6 4$ </td><td>300</td><td>-1.85</td><td>no</td></tr></table>

Table 4: Anchor-sensitivity of the L2-positive-at-the-anchor verdict (all 27 confirmatory cells). A cell counts iff mean ITE ≥ anchor AND CI lower $> 0$ (CI condition holds in every cell). Frozen anchor 5.371; replicated anchor dose 4.65. Exactly three cells sit inside the band and flip; every other cell is negative under both.
<table><tr><td>Cell (study, victim, €)</td><td>mean</td><td>CI low</td><td> $\geq 5 . 3 7 1$ </td><td>≥4.65</td></tr><tr><td>B: TeCoA@64</td><td>5.24</td><td>4.74</td><td>no</td><td>yes flip</td></tr><tr><td>C: TeCoA@64</td><td>4.98</td><td>4.35</td><td>no</td><td>yes flip</td></tr><tr><td>C: PMG@64</td><td>4.90</td><td>4.42</td><td>no</td><td>yes fip</td></tr><tr><td>C: TRADES @64</td><td>4.30</td><td>3.85</td><td>no</td><td>no</td></tr><tr><td colspan="5">All remaining 23 cells: mean  $\mathrm { I T E } < 4 . 0 ,$  negative under both anchors</td></tr></table>

## 5.3 Recipe comparison

The preregistered recipe could in principle sit far below the achievable optimization frontier, reducing the boundary claim to a statement about one optimizer. We ran targeted PGD and C&W-style attacks (100 iterations each) alongside the preregistered recipe on exploration-only data, under the identical cohort, anchors, shuffles, and judge. No variant crosses $S _ { \mathrm { m i n } }$ in any cell; the preregistered recipe itself achieves the highest ITE in five of six victim-budget combinations. PGD-100 attains higher victim-side alignment (VTS up to 0.63 vs. 0.49) while transferring equal or less evidence. This is a within-optimization replication of the L1⊥L2 dissociation: pushing the victim’s target score harder does not buy more external evidence (Table 8).

## 5.4 Within-encoder and cross-encoder diagnosticity

Within every robust-encoder cell, per-sample alignment gain predicts evidence gain $( \rho = 0 . 2 4 \ – 0 . 5 1 ;$ Fisher 95% CIs exclude zero, e.g. TeCoA@16 [0.42, 0.59]; stable across mapping seeds). Within vanilla CLIP, the observed $\rho \in \left[ - 0 . 0 4 , 0 . 0 0 \right]$ carry Fisher 95% CIs no wider than $[ - 0 . 1 5 , 0 . 1 1 ]$ . A post-hoc TOST against an exploratory small-association tolerance $| \rho | \ge 0 . 1 5$ (conventional bound; post-confirmatory, independent-sample approximation, clustering not accounted for) gives $p = 0 . 0 2 8 / 0 . 0 1 0 / 0 . 0 0 5$ at ϵ = $4 / 1 6 / 6 4 ,$ , consistent with but not proving a negligible association. The TOST covers vanilla only. Between victims in Study B, the mean-VTS ranking inverts the mean-ITE ranking $( \rho = - 1 . 0 ; n { = } 3$ , descriptive):

![](images/165a12106a71ad196d337f2ce1330fd5e0519dd1f0d803de10b6637bf0c54ee6.jpg)  
Figure 3: Within-encoder slopes at $\scriptstyle \epsilon = 6 4 / 2 5 5$ across six encoders (Study-C cohort, $n { = } 3 0 0$ per panel). X-axis ranges vary; line angles are not directly comparable.

vanilla has the highest post-attack alignment and the lowest evidence. Pooling the nine cells gives $\rho \approx + 0 . 0 8 ;$ the two levels cancel.

## 5.5 Study C: additional encoders

A separately frozen design extends the grid to PMG, TRADES, and SigLIP on a six-encoder intersection cohort (Table 7 in the appendix).

H C1: robustness extension. All four Holm-corrected contrasts reject at the bootstrap floor. PMG−vanilla: +2.53/+4.24; TRADES−vanilla: +2.33/+3.64 ITE units at $\scriptstyle \epsilon = 1 6 / 6 4$ . The gap between robust and naive now replicates on four distinct recipes, whose ϵ=64 cells cluster tightly (ITE 3.3–5.0).

H C2: family extension (descriptive). SigLIP remains at vanilla-like levels at every budget (ITE 0.08–0.35). SigLIP differs from the CLIP encoders in objective, scale, and training recipe; this comparison does not isolate the cause of the robust-naive gap. The four-recipe robust-versus-naive contrast is the strongest available, but it remains a checkpoint comparison rather than a matched training intervention.

H C3: cross-encoder anti-ordering (preregistered failure). The direction-only prediction that victimmean VTS anti-ranks ITE does not survive at $n { = } 6 ~ ( \rho ~ = ~ + 0 . 0 3 , p ~ = ~ 0 . 9 6$ , computed at ϵ=64 on the Study-C cohort): SigLIP has both the lowest VTS and the lowest ITE. The Study-B inversion is real but conditional. Across all six encoders, we find no monotone alignment–evidence relation in either direction. This strengthens, rather than weakens, the practical conclusion: cross-encoder VTS comparison fails in a way that includes inversion between regimes but is not reducible to it.

## 5.6 Secondary observations

ITR (anchor in top-5) corroborates the ordering: TeCoA@64 0.293, PMG@64 0.263, TRADES@64 0.213, FARE@64 0.180, vanilla@64 0.013, SigLIP@64 0.013. That nearly one in three test images of TeCoA@64 has the attack-chosen target class in the independent judge’s top-5 confirms that the ITE signal is not a statistical artifact, even though the mean does not reach the reference level. The secondary ViT-B/16 judge corroborates direction and ordering at ϵ=16 and 64, with per-sample cross-judge Spearman $\rho = 0 . 5 5 \mathrm { - } 0 . 5 9$ among robust victims. At ϵ=4 it shows no robust-victim elevation; those cells are descriptive in the frozen design. Cross-judge logit scales are uncalibrated, so the ViT judge is not compared to the R50-derived reference level. Low vanilla cross-judge agreement $( \rho \approx 0 . 2 0 \ – 0 . 2 5 )$ reflects the near-zero signal range.

## 6 Discussion

What VTS measures and does not measure. Within a robustly fine-tuned encoder, VTS tracks independent evidence per sample. Within vanilla CLIP, the monotone association is consistent with zero under a post-hoc approximation. Across encoders, the naive-versus-robust inversion does not extend to SigLIP. These three facts together limit VTS to a within-encoder diagnostic for robust checkpoints. To quantify the cross-encoder prediction question, we ran a post-hoc sensitivity check (not preregistered; class-grouped 5-fold crossvalidation over 8,100 pooled records): adding ∆VTS to an additive encoder-plus-budget baseline changes top-5-recovery AUC from 0.829 to 0.839 and $R ^ { 2 }$ from 0.236 to 0.288. These are small positive increments under this specific baseline, split, and set of endpoints. The folds hold out source classes, not encoders, so the result applies to seen encoders and does not establish or rule out incremental value in other settings.

Table 5: Study B (confirmatory): independent target evidence $\mathrm { I T E } _ { S _ { a } }$ per cell, pooled over mapping seeds 101/202/303 $\scriptstyle ( n = 3 0 0 / \mathrm { c e l l } ;$ ; R50 judge; held-out ImageNet-100 rows). Rule: L2-positive at the anchor ⇐⇒ mean $\geq S _ { \mathrm { m i n } } = 5 . 3 7 1$ and class-cluster bootstrap 95% CI lower bound > 0. ITR = anchor-in-top-5 rate. VTS = victim-space target similarity after attack (L1 diagnostic).
<table><tr><td>Victim</td><td>€</td><td>n</td><td>ITE mean</td><td>ITE sd</td><td> $\mathrm { C I _ { 9 5 } }$  lower</td><td>ITR top-5</td><td>VTS</td></tr><tr><td>vanilla</td><td>4/255</td><td>300</td><td>0.1653</td><td>0.6131</td><td>0.1015</td><td>0.0033</td><td>0.4298</td></tr><tr><td>vanilla</td><td>16/255</td><td>300</td><td>0.611</td><td>1.6177</td><td>0.4464</td><td>0.01</td><td>0.4784</td></tr><tr><td>vanilla</td><td>64/255</td><td>300</td><td>0.7935</td><td>1.9026</td><td>0.5988</td><td>0.0133</td><td>0.4879</td></tr><tr><td>FARE</td><td>4/255</td><td>300</td><td>0.3117</td><td>0.8091</td><td>0.2361</td><td>0.0067</td><td>0.2362</td></tr><tr><td>FARE</td><td>16/255</td><td>300</td><td>2.4412</td><td>2.7311</td><td>2.1131</td><td>0.08</td><td>0.3499</td></tr><tr><td>FARE</td><td>64/255</td><td>300</td><td>3.3974</td><td>3.2656</td><td>3.025</td><td>0.19</td><td>0.3842</td></tr><tr><td>TeCoA</td><td>4/255</td><td>300</td><td>0.2903</td><td>0.6925</td><td>0.2191</td><td>0.0033</td><td>0.1871</td></tr><tr><td>TeCoA</td><td>16/255</td><td>300</td><td>3.162</td><td>3.3292</td><td>2.781</td><td>0.1233</td><td>0.2778</td></tr><tr><td>TeCoA</td><td>64/255</td><td>300</td><td>5.2441</td><td>4.4289</td><td>4.7433</td><td>0.2933</td><td>0.3222</td></tr></table>

† inconclusive-at-margin: CI condition passes, mean within 0.5 of $S _ { \mathrm { m i n } }$ (see Sensitivity). No cell is L2-positive at the anchor.

Table 6: Two-level diagnosticity of victim-space alignment (confirmatory). Within-cell: Spearman $\rho$ between per-sample ∆VTS and $\mathrm { I T E } _ { S _ { a } }$ (all permutation $p \leq 0 . 0 0 0 5$ except vanilla, ns). Between-victim: victimmean VTS vs victim-mean ITE. Pooled cell-level $\rho$ shown to exhibit the Simpson cancellation that pooled correlations conceal.
<table><tr><td>Victim</td><td> $\rho \left( \epsilon { = } 4 \right)$ </td><td> $\rho \left( \epsilon { = } 1 6 \right)$ </td><td> $\rho \left( \epsilon { = } 6 4 \right)$ </td></tr><tr><td>vanilla</td><td>-0.041</td><td>-0.016</td><td>-0.001</td></tr><tr><td>FARE</td><td>+0.314</td><td>+0.368</td><td>+0.236</td></tr><tr><td>TeCoA</td><td>+0.313</td><td>+0.512</td><td>+0.409</td></tr><tr><td>Between-victim means (van/FARE/TeCoA):</td><td></td><td></td><td></td></tr><tr><td>VTS 0.465/0.323/0.262 vs ITE 0.52/2.05/2.90;</td><td></td><td></td><td></td></tr><tr><td> $\rho = - 1 . 0 ( n { = } 3 ,$  descriptive) Pooled cell-level (9 cells):  $\rho = + 0 . 0 8 3$  (cancellation)</td><td></td><td></td><td></td></tr></table>

The robustness–evidence gap. Two findings sit side by side. Robust recipes show higher mean ITE than naive ones: large effects, monotone in budget, Holm-corrected. Within robust cells, alignment gain correlates with evidence gain; the correlation itself is not monotone in budget (FARE peaks at ϵ=16). Neither finding is a training-causal claim, because the encoders are released checkpoints rather than matched interventions. No cell’s mean reaches the reference level, even at $\epsilon { = } 6 4 / 2 5 5$ , an unusually large budget. The evaluation budgets exceed the reported TeCoA training budget; FARE’s exact training budget is unresolved. The contrasts are budget-matched, so the robust-naive gap is not explained by budget differences, but recipe attribution and interpretation relative to training budgets require verified provenance.

Near-threshold cells. The three best cells sit between the two anchor calibrations (5.371 frozen, 4.65 replicated). We report them as inconclusive rather than resolved in either direction. The anchor’s calibration uncertainty (approximately 15% between draws) is comparable to the gap between these cells and the reference level, which is exactly the situation where a binary verdict would be misleading.

Table 7: Study C per-cell results by budget (confirmatory; n=300/cell; R50 judge; cf. Holm-4 and H C2/H C3 in the main text). No cell is L2-positive at the anchor.
<table><tr><td>€</td><td>Metric</td><td>vanilla</td><td>FARE</td><td>TeCoA</td><td>PMG</td><td>TRADES</td><td>SigLIP</td></tr><tr><td>4/255</td><td>ITE mean</td><td>0.1961</td><td>0.3288</td><td>0.2786</td><td>0.3893</td><td>0.3748</td><td>0.0833</td></tr><tr><td></td><td>CI95 lower</td><td>0.1288</td><td>0.2417</td><td>0.1987</td><td>0.3094</td><td>0.2954</td><td>0.0196</td></tr><tr><td></td><td>ITR top-5</td><td>0.01</td><td>0.0033</td><td>0.0067</td><td>0.01</td><td>0.0033</td><td>0.0</td></tr><tr><td></td><td>VTS</td><td>0.4325</td><td>0.2342</td><td>0.1846</td><td>0.2032</td><td>0.2075</td><td>0.2367</td></tr><tr><td>16/255</td><td>ITE mean</td><td>0.5428</td><td>2.3106</td><td>2.9692</td><td>3.0684</td><td>2.8698</td><td>0.2907</td></tr><tr><td></td><td>CI95 lower</td><td>0.3955</td><td>1.9795</td><td>2.5086</td><td>2.6842</td><td>2.4941</td><td>0.1533</td></tr><tr><td></td><td>ITR top-5</td><td>0.02</td><td>0.0833</td><td>0.14</td><td>0.13</td><td>0.1333</td><td>0.01</td></tr><tr><td></td><td>VTS</td><td>0.4811</td><td>0.3478</td><td>0.2757</td><td>0.2845</td><td>0.2873</td><td>0.2788</td></tr><tr><td>64/255</td><td>ITE mean</td><td>0.6595</td><td>3.2891</td><td>4.9815</td><td>4.8988</td><td>4.302</td><td>0.3493</td></tr><tr><td></td><td>CI95 lower</td><td>0.497</td><td>2.8931</td><td>4.3486</td><td>4.4232</td><td>3.8512</td><td>0.1969</td></tr><tr><td></td><td>ITR top-5</td><td>0.02</td><td>0.18</td><td>0.2667</td><td>0.2633</td><td>0.2133</td><td>0.0133</td></tr><tr><td></td><td>VTS</td><td>0.4913</td><td>0.3827</td><td>0.3207</td><td>0.3213</td><td>0.3182</td><td>0.2801</td></tr></table>

Table 8: Recipe-frontier check (exploratory split; identical cohort, anchors, and judge to the calibration pilot). ITE under the preregistered recipe (frozen-20) vs standard stronger optimizers at 100 iterations. No variant crosses $S _ { \mathrm { m i n } } = 5 . 3 7 1$ in any cell; the preregistered recipe attains the best ITE in five of six victim–budget combinations. PGD-100 attains higher victim-side alignment (VTS) than the recipe while transferring less evidence.
<table><tr><td rowspan="2">Victim</td><td rowspan="2">€</td><td colspan="3">ITE mean</td><td colspan="3">VTS</td></tr><tr><td>frozen</td><td>PGD-100</td><td>C&amp;W-100</td><td>frozen</td><td>PGD-100</td><td>C&amp;W-100</td></tr><tr><td>vanilla</td><td>16/255</td><td>0.6571</td><td>0.3098</td><td>0.3626</td><td>0.4789</td><td>0.5723</td><td>0.4898</td></tr><tr><td>vanilla</td><td>64/255</td><td>0.7479</td><td>0.6111</td><td>0.5009</td><td>0.4883</td><td>0.63</td><td>0.5029</td></tr><tr><td>FARE</td><td>16/255</td><td>2.569</td><td>2.4876</td><td>2.0176</td><td>0.3483</td><td>0.4254</td><td>0.3705</td></tr><tr><td>FARE</td><td>64/255</td><td>3.3824</td><td>2.991</td><td>1.5034</td><td>0.383</td><td>0.5423</td><td>0.4411</td></tr><tr><td>TeCoA</td><td>16/255</td><td>3.1555</td><td>3.4647</td><td>3.2887</td><td>0.2762</td><td>0.3296</td><td>0.3</td></tr><tr><td>TeCoA</td><td>64/255</td><td>5.0246</td><td>4.5935</td><td>3.0152</td><td>0.3198</td><td>0.4508</td><td>0.3839</td></tr></table>

A reporting protocol for adversarial semantic claims. No existing standard covers what an adversarial “semantic” claim needs to report. In place of raw VTS, we recommend: (i) scope alignment claims to the evaluated encoder; (ii) report at least one validated independent endpoint, with graded real-target positive controls, shuffled-target negatives, and a judge competence check; (iii) state the optimization recipe and its distance to standard stronger attacks; (iv) for robust encoders, evaluate at and just beyond the training budget. These four items distinguish the calibration protocol from prior transfer-evaluation approaches.

Checkpoint provenance. Our FARE checkpoint is labeled $\epsilon _ { \mathrm { t r a i n } } { = } 4 / 2 5 5$ based on the artifact-repository README, but the RobustVLM release lists multiple FARE variants for $\mathrm { V i T - B } / 3 2 ( \epsilon _ { \mathrm { t r a i n } } { = } 1 $ and 4/255), and we have not independently verified which upstream checkpoint produced our local file. The checkpoint’s SHA-256 hash matches the frozen manifest, so the experiments are internally consistent, but the trainingbudget attribution and the regime-mismatch interpretation depend on resolving this provenance question. TeCoA’s 1/255 budget is confirmed from its paper.

What the protocol adds. The standard evaluation in this space reports attack success rate or victimside alignment. Our instrument adds three things: graded real-target blend controls that establish the judge’s sensitivity to actual target evidence; shuffled-target negatives that correct for non-specific logit-mass movement; and a pre-specified reference level that anchors the verdict to a measurable quantity rather than a threshold chosen post hoc. Any of these can be adopted independently of the others.

Mechanism. The observed pattern is consistent with robust-features and off-manifold accounts [2, 18, 28, 33, 34], but no mechanism quantity was measured confirmatorily. Identifying the causal pathway from adversarial training to independent evidence transfer would require matched training interventions or a continuous robustness axis, which are outside the scope of this paper.

Limitations. Six encoders (five ViT-B/32-family recipes plus SigLIP), one dataset (ImageNet-100), one attack objective (fixed-target), three budgets, one preregistered recipe. The preregistered cross-encoder antiordering hypothesis failed (ρ ≈ 0); the between-encoder claim is therefore stated conditionally. Fine-grained and texture datasets were not run (classes outside the judge’s label space). Encoders are released checkpoints, not matched training interventions; FARE’s training budget is unresolved. Claims are capped at L2 under supervised ImageNet-1k judges (no L3 or L4 endpoint; the secondary judge shares the ViT family). The reference level inherits approximately 15% inter-draw uncertainty. No mechanism quantity was measured confirmatorily. The adaptive-objective arm was dropped by its own preregistered gate.

Future work. Several extensions follow from these findings. Matched training interventions that vary adversarial-training strength along a continuum would isolate whether robustness itself causes the alignment– evidence gap or whether the gap tracks a correlated property of the released checkpoints. Extending the instrument to additional judge paradigms (self-supervised probes, zero-shot captioning models) would test whether the L2 boundary is judge-specific or a property of the encoders themselves. Downstream propagation (L3) remains untested: if the classifier-level evidence we measure propagates to generative or retrieval endpoints, the sub-threshold finding may understate the practical risk. The reporting protocol could be developed into a community benchmark covering additional attack families, encoder architectures, and perturbation norms. A human-grounded study (L4) would determine whether the blend-derived reference level corresponds to any perceptual threshold.

## References

[1] Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In Proceedings ofthe 35th International Conference on Machine Learning (ICML), pages 274–283, 2018. URL https://arxiv.org/abs/1802. 00420.

[2] Mohamed Awad, Mahmoud Mohamed, and Walid Gomaa. Defense that attacks: How robust models become better attackers. In Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP 2026), volume 2, pages 539–549, 2026. URL https://doi. org/10.5220/0014231700004084.

[3] Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, et al. Measuring what matters: Construct validity in large language model benchmarks. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2025. URL https://arxiv.org/abs/2511. 04703.

[4] Matan Ben-Tov et al. Universal jailbreak suffixes are strong attention hijackers. Transactions of the Association for Computational Linguistics (TACL), 14:1028–1050, 2026. URL https://arxiv. org/abs/2506.12880.

[5] Anand Bhattad, Min Jin Chong, Kaizhao Liang, Bo Li, and David A. Forsyth. Unrestricted adversarial examples via semantic manipulation. In International Conference on Learning Representations (ICLR), 2020. URL https://arxiv.org/abs/1904.06347.

[6] Hongyu Chen and Seraphina Goldfarb-Tarrant. Safer or luckier? LLMs as safety evaluators are not robust to artifacts. In Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2025. URL https://arxiv.org/abs/2503.09347.

[7] Antonio Emanuele Cina et al. Evaluating the evaluators: Trust in adversarial robustness tests.\` arXiv preprint arXiv:2507.03450, 2025. URL https://arxiv.org/abs/2507.03450.

[8] Francesco Croce, Christian Schlarmann, Naman Deep Singh, and Matthias Hein. Adversarially robust CLIP models can induce better (robust) perceptual metrics. arXiv preprint arXiv:2502.11725, 2025. URL https://arxiv.org/abs/2502.11725.

[9] Lee J. Cronbach and Paul E. Meehl. Construct validity in psychological tests. Psychological Bulletin, 52(4):281–302, 1955. URL https://doi.org/10.1037/h0040957.

[10] Gamaleldin F. Elsayed, Shreya Shankar, Brian Cheung, Nicolas Papernot, Alexey Kurakin, Ian Goodfellow, and Jascha Sohl-Dickstein. Adversarial examples that fool both computer vision and timelimited humans. In Advances in Neural Information Processing Systems (NeurIPS), 2018. URL https://arxiv.org/abs/1802.08195.

[11] Hao Fang et al. One perturbation is enough: On generating universal adversarial perturbations against vision-language pre-training models. In IEEE/CVF International Conference on Computer Vision (ICCV), 2025. URL https://arxiv.org/abs/2406.05491.

[12] Dren Fazlija, Monty-Maximilian Zuhlke, Johanna Schrader, Arkadij Orlov, Clara Stein, Iyiola E.¨ Olatunji, and Daniel Kudenko. Scooter: A human evaluation framework for unrestricted adversarial examples. arXiv preprint arXiv:2507.07776, 2025. URL https://arxiv.org/abs/2507. 07776.

[13] Timo Freiesleben. Establishing construct validity in llm capability benchmarks requires nomological networks. arXiv preprint arXiv:2603.15121, 2026. URL https://arxiv.org/abs/2603.15121.

[14] Roy Ganz and Michael Elad. CLIPAG: Towards generator-free text-to-image generation. In IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024. URL https://arxiv.org/ abs/2306.16805.

[15] Roy Ganz, Bahjat Kawar, and Michael Elad. Do perceptually aligned gradients imply robustness? In International Conference on Machine Learning (ICML), 2023. URL https://arxiv.org/abs/ 2207.11378.

[16] Hanxun Huang et al. X-Transfer Attacks: Towards super transferable adversarial attacks on CLIP. In International Conference on Machine Learning (ICML), 2025. URL https://arxiv.org/abs/ 2505.05528.

[17] Ruixuan Huang et al. Guidedbench: Measuring and mitigating the evaluation discrepancies of in-thewild LLM jailbreak methods. arXiv preprint arXiv:2502.16903, 2025. URL https://arxiv.org/ abs/2502.16903.

[18] Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Logan Engstrom, Brandon Tran, and Aleksander Madry. Adversarial examples are not bugs, they are features. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1905.02175.

[19] Simran Kaur, Jeremy Cohen, and Zachary C. Lipton. Are perceptually-aligned gradients a general property of robust classifiers? arXiv preprint arXiv:1910.08640, 2019. URL https://arxiv.org/ abs/1910.08640.

[20] Meir Yossef Levi and Guy Gilboa. The double-ellipsoid geometry of CLIP. In International Conference on Machine Learning (ICML), 2025. URL https://arxiv.org/abs/2411.14517.

[21] Shuai Li, Xiaoyu Jiang, and Xiaoguang Ma. Transcending adversarial perturbations: Manifoldaided adversarial examples with legitimate semantics. arXiv preprint arXiv:2402.03095, 2024. URL https://arxiv.org/abs/2402.03095.

[22] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2203.02053.

[23] Peter Lorenz, Dominik Strassel, Margret Keuper, and Janis Keuper. Is robustbench/autoattack a suitable benchmark for adversarial robustness? arXiv preprint arXiv:2112.01601, 2021. URL https: //arxiv.org/abs/2112.01601.

[24] Chengzhi Mao, Scott Geng, Junfeng Yang, Xin Wang, and Carl Vondrick. Understanding zero-shot adversarial robustness for large-scale models. In International Conference on Learning Representations (ICLR), 2023. URL https://arxiv.org/abs/2212.07016.

[25] Samuel Messick. Validity of psychological assessment: Validation of inferences from persons’ responses and performances as scientific inquiry into score meaning. American Psychologist, 50(9):741–749, 1995. URL https://doi.org/10.1037/0003-066X.50.9.741.

[26] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning (ICML), pages 8748–8763, 2021. URL https: //arxiv.org/abs/2103.00020.

[27] Olawale Salaudeen, Anka Reuel, Ahmed Ahmed, Suhana Bedi, Zachary Robertson, Sudharsan Sundar, Ben Domingue, Angelina Wang, and Sanmi Koyejo. Measurement to meaning: A validity-centered framework for ai evaluation. arXiv preprint arXiv:2505.10573, 2025. URL https://arxiv.org/ abs/2505.10573.

[28] Shibani Santurkar, Dimitris Tsipras, Brandon Tran, Andrew Ilyas, Logan Engstrom, and Aleksander Madry. Image synthesis with a single (robust) classifier. In Advances in Neural Information Processing Systems (NeurIPS), 2019. URL https://arxiv.org/abs/1906.09453.

[29] Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein. Robust CLIP: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models.

In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. URL https://arxiv.org/abs/2402.12336.

[30] Leo Schwinn et al. A coin flip for safety: LLM judges fail to reliably measure adversarial robustness. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/ 2603.06594.

[31] Guobin Shen et al. Pandaguard: Systematic evaluation of LLM safety against jailbreaking attacks. arXiv preprint arXiv:2505.13862, 2025. URL https://arxiv.org/abs/2505.13862.

[32] Jacob Mitchell Springer, Melanie Mitchell, and Garrett T. Kenyon. A little robustness goes a long way: Leveraging robust features for targeted transfer attacks. In Advances in Neural Information Processing Systems (NeurIPS), 2021. URL https://arxiv.org/abs/2106.02105.

[33] Suraj Srinivas, Sebastian Bordt, and Hima Lakkaraju. Which models have perceptually-aligned gradients? an explanation via off-manifold robustness. In Advances in Neural Information Processing Systems (NeurIPS), 2023. URL https://arxiv.org/abs/2305.19101.

[34] David Stutz, Matthias Hein, and Bernt Schiele. Disentangling adversarial robustness and generalization. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019. URL https: //arxiv.org/abs/1812.00740.

[35] Vijay Veerabadran, Josh Goldman, Shreya Shankar, Brian Cheung, Nicolas Papernot, Alexey Kurakin, Ian Goodfellow, Jonathon Shlens, Jascha Sohl-Dickstein, Michael C. Mozer, and Gamaleldin F. Elsayed. Subtle adversarial image manipulations influence both human and machine perception. Nature Commu nications, 14:4933, 2023. URL https://doi.org/10.1038/s41467-023-40499-0.

[36] Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. Universal adversarial triggers for attacking and analyzing NLP. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), pages 2153–2162, 2019. URL https: //arxiv.org/abs/1908.07125.

[37] Sibo Wang, Jie Zhang, Zheng Yuan, and Shiguang Shan. Pre-trained model guided fine-tuning for zero-shot adversarial robustness. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. URL https://arxiv.org/abs/2401.04350.

[38] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023. URL https://arxiv.org/abs/2303.15343.

[39] Andi Zhang, Xuan Ding, Steven McDonagh, and Samuel Kaski. Concept-based adversarial attack: A probabilistic perspective. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2507.02965.

[40] Hongyang Zhang, Yaodong Yu, Jiantao Jiao, Eric P. Xing, Laurent El Ghaoui, and Michael I. Jordan. Theoretically principled trade-off between robustness and accuracy. In Proceedings of the 36th International Conference on Machine Learning (ICML), pages 7472–7482, 2019. URL https://arxiv.org/abs/1901.08573.

[41] Jiaming Zhang et al. Anyattack: Towards large-scale self-supervised adversarial attacks on visionlanguage models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. URL https://arxiv.org/abs/2410.05346.

[42] Tingwei Zhang et al. Adversarial illusions in multi-modal embeddings. In USENIX Security Symposium, 2024. URL https://www.usenix.org/system/files/ usenixsecurity24-zhang-tingwei.pdf.

[43] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043, 2023. URL https://arxiv.org/abs/2307.15043.