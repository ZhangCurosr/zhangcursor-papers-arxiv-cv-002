# How Much Physics Can Vision Learn from Audio? Probing and Distilling Damping Structure

Zainab Afolabi Amazon Web Services zafolabi@amazon.co.uk

Vadim Selyutin \* Amazon Web Services seliutin@amazon.ae

Babs Khalidson \* Amazon Web Services khalbabs@amazon.co.uk

Zhao Gao \* Amazon Web Services orangez@amazon.co.uk

Mohamed Ghoneim Amazon Web Services ghoneimm@amazon.co.uk

## Abstract

Physical understanding is crucial for decision making, yet vision models struggle to infer properties such as damping that govern how materials behave. Audio encodes such properties directly, because an object’s acoustic response to impact reveals its damping through measurable decay. We probe frozen encoders for the quality factor Q, a dimensionless measure of how many oscillations an object completes before its ring dies away, and report how well each embedding orders impacts by Q using the Spearman rank correlation ρ, for which $\rho = 1$ is a perfect ordering and $\rho = 0$ an unrelated one. Audio embeddings reach $\rho = 0 . 7 3$ and retain most of that advantage on materials withheld from training, while a single video frame taken at the moment of impact, encoded by CLIP, supports only $\rho = 0 . 2 8$ . Having established where the signal lives, we ask how much of that gap vision can recover. We distill a frozen audio teacher into a frozen CLIP backbone through a trained projection head and find that this recovers a useful fraction of the audio advantage. Because pooled correlations on this task are dominated in practice by category identity, we also evaluate under a leave-one-material-out protocol in which the held-out material is withheld from the projection head as well as the probe. Distillation with a four-frame input improves the correlation there from 0.234 to 0.338 (+0.053 over the matched four-frame baseline, 14 of 17 materials), a gain that no category-level strategy can produce. Widening the visual input to four frames spanning the impact supplies half of that improvement at no acoustic cost, which indicates that part of what a single frame misses is visible dynamics.

## 1 Introduction

A robot clearing a table observes two cups that look alike, one ceramic and one plastic. A vision model might correctly label their materials, but this categorical distinction does not reveal how they behave, since the ceramic cup shatters when dropped while the plastic cup bounces. Without physical grounding, vision cannot predict these divergent futures.

This appearance–reality gap pervades modern computer vision. Large vision models achieve strong recognition performance, but what they encode is how things look rather than how things behave. Physics benchmarks confirm this gap. PhysBench [Chow et al., 2025] found that GPT-4o achieves only 46.99% on physical dynamics (49.49% overall, against 95.87% for humans), and Physion++ [Tung et al., 2023] showed that all models remain well below human performance on latent properties such as mass, friction, or elasticity.

Audio captures what appearance does not. An object’s acoustic response to impact is governed by physical laws: stiffness, density, and geometry set the resonant frequency, and internal friction, radiation, and mounting set the damping. We confirm the gap directly. Frozen audio embeddings predict the quality factor $Q$ at $\rho = 0 . 7 3$ , while a single CLIP-encoded video frame taken at the moment of impact supports only $\rho ~ = ~ 0 . 2 8$ Here $Q$ is a dimensionless measure of how many oscillations a struck object completes before its ring dies away, so a highly resonant glass has a large $Q$ and a heavily damped cloth a small one. We write $\rho$ for the Spearman rank correlation between predicted and true values: $\rho = 1$ means the model orders impacts by $Q$ perfectly, and $\rho = 0$ means its ordering carries no information about the truth. We use $\rho$ throughout because we care about whether a model can rank objects by how much they ring, not about the absolute scale of its predictions. Our question is the narrower one of how far a frozen visual encoder can be pushed toward the audio number.

Our answer is that it can be pushed a useful fraction of the way. We distill from a frozen audio teacher into a frozen CLIP encoder, training a projection head as the only new component, and measure the outcome under a protocol in which knowing the material category earns nothing: correlations are computed within a material that was held out from the projection head and from the probe alike. Distillation improves prediction there for most of the 17 materials. Part of this improvement needs no audio at all, since giving the encoder four frames spanning the impact, so that it can see the object deform and recoil, supplies about half. Most of the audio signal nonetheless stays out of reach, which we take to mark the interesting part of the problem rather than a settled question.

## 1.1 Contributions

• We propose probing frozen encoders for a continuous physical property, the quality factor $Q .$ , rather than for the categorical material labels used in prior evaluations. Our measurements show that audio embeddings predict $Q$ at $\rho ~ = ~ 0 . 7 3$ whereas a single CLIP frame reaches 0.28, that this audio advantage persists for materials withheld from training $( \rho \approx 0 . 6 0 )$ ), and that every visual encoder we test is near chance at ranking objects within wood, ceramic, and plastic and reaches at most 0.36 within metal.

• We show that part of the visual shortfall is temporal rather than material. Four frames spanning the impact raise vision-only prediction on unseen materials by 0.051 without any acoustic supervision, which suggests that some of what a single frame misses is visible dynamics rather than hidden material structure.

• We propose distilling a frozen audio teacher into a frozen visual encoder through a trained projection head, and our experiments demonstrate that this moves a useful fraction of the signal across. The distilled CLIP encoder reaches $\rho = 0 . 4 4$ on $Q$ and, with a four-frame input, improves prediction from 0.234 to 0.338 on materials withheld from the head as well as the probe (+0.104, 15 of 17 materials, $p = 0 . 0 0 2$ ; +0.053 over the matched four-frame baseline, 14 of 17 materials).

• We report controls that establish where this gain comes from. A capacity ablation, a shuffled-teacher control, and a material-label reference together attribute most of the improvement to acoustic correspondence rather than to added parameters or category recognition. We also document the experimental choices that inflate such measurements when these controls are absent.

## 2 Related Work

Impact sounds encode material properties that appearance cannot reveal. Owens et al. [2016a] es tablished this connection with the Greatest Hits dataset, demonstrating that visual models can learn to predict impact sounds. RealImpact [Clarke et al., 2023] extends this with force profiles and calibrated measurements. ObjectFolder [Gao et al., 2023] quantifies the relative contributions of vision, audio, and touch for material understanding, finding that audio often provides complementary information. In robotics, audio frequently outperforms vision for physical property estimation: Thankaraj and Pinto [2023] show that auditory feedback improves manipulation, and SonicSense [Liu and Chen, 2024] demonstrates object perception through acoustic vibration. We connect this evidence to representation learning: if audio encodes physics, can this structure be distilled into vision?

Joint audio-visual embeddings have evolved from contrastive methods [Arandjelovic and Zisserman,´ 2017, Owens and Efros, 2018] to masked autoencoders [Gong et al., 2023, Huang et al., 2023]. PE-AV [Vyas et al., 2025] scales to 100M pairs, achieving state-of-the-art results on audio-visual retrieval. ImageBind [Girdhar et al., 2023] aligns six modalities through image-paired data. Prior work evaluates these models on retrieval and classification, but, to our knowledge, none probes what frozen audio-visual embeddings capture about physical properties, or whether this knowledge can transfer to vision alone.

Recent benchmarks reveal systematic failures in physical reasoning. Beyond PhysBench [Chow et al., 2025], Physion [Bear et al., 2021] shows that models fall short of humans at predicting physical outcomes, Physion++ [Tung et al., 2023] extends this to outcomes that depend on latent properties, and CLEVRER [Yi et al., 2020] and IntPhys [Riochet et al., 2021] probe causal and intuitive physics understanding. These benchmarks focus on dynamics prediction, and we complement them by probing continuous material properties.

Audio and vision have long supervised one another: ambient sound can train visual representations [Owens et al., 2016b], and visual networks can train sound representations [Aytar et al., 2016]. Knowledge distillation transfers representations across architectures [Hinton et al., 2015] or modalities: Gupta et al. [2016] transferred supervision from labeled RGB images to depth and opticalflow representations. Our work differs in targeting specific physically relevant structure rather than general representations, and takes a step toward making acoustically grounded properties partially accessible to vision alone.

## 3 The Quality Factor: A Continuous Physical Property

## 3.1 Why Q?

The quality factor $Q = \pi f _ { 0 } / \beta$ combines resonant frequency $( f _ { 0 } )$ and damping rate (β) into one dimensionless scalar. The damping rate $\beta$ sets how quickly a ringing object falls silent in absolute time, whereas $Q$ counts how many oscillation cycles it completes before decaying, so a high-pitched glass can fall silent sooner than a lower-pitched metal bowl while having the higher $Q$

We target this compound quantity for three reasons. Being dimensionless, $Q$ is comparable across objects regardless of size or pitch, so a wine glass and a steel beam can share a Q despite frequencies orders of magnitude apart. It also varies continuously within material categories: a thin metal sheet and a solid metal block are both “metal” yet differ substantially in $Q ,$ so predicting it demands more than category recognition. Finally, it has direct physical consequences, determining whether an object rings, bounces, or absorbs energy on impact. These properties make $Q$ a stricter test of physical grounding than frequency, which is largely predictable from visible geometry, or the damping rate alone, which is confounded with object size.

## 3.2 Extraction Pipeline

We extract Q from audio recordings in the Greatest Hits dataset [Owens et al., 2016a], which contains 977 videos of objects being hit with a drumstick. Our copy carries annotations for 975, totaling 28,444 individual impacts across 17 material categories.

Figure 1 illustrates both the quantity and the method. For each annotated impact, we take the magnitude of the analytic signal (the Hilbert envelope) to obtain an amplitude envelope $A ( t )$ and fit the decay model $A ( t ) = A _ { 0 } e ^ { - \beta t } + C$ by nonlinear least squares, with $\beta$ bounded to $[ 0 . \dot { 1 } , 1 0 0 0 ] ~ \mathrm { s } ^ { - 1 }$ and the additive constant $C$ absorbing background noise. The resonant frequency $f _ { 0 }$ is the dominant peak of the power spectrum above 50 Hz, a floor that avoids low-frequency room noise. Retaining only fits with $R ^ { 2 } \ge 0 . 8$ leaves 17,958 usable impacts, 63% of the total. Rejections concentrate on granular and scattering surfaces, where the response has no single resonant mode to fit: water retains 24% of its annotated impacts, gravel 33%, and dirt 36%, against 83% for cloth and 91% for glass. We return to this selection effect in Section 8.

## 3.3 Validation

The extracted values match physical expectation. Median $Q$ rises monotonically from the soft, absorptive materials to the resonant ones: cloth 15, carpet 23, wood 55, plastic 89, metal 134, ceramic

![](images/402907b89e6c0238ec84104241a07f3dbb7b5dee68b0606d88d6116b303d20e7.jpg)

![](images/1540aba893dbdb91f44c3f9b4087eed0de99ac6bd172fb95e9bc949b0d6c07ca.jpg)  
Figure 1: What Q measures, and how we transfer it. (a) Example impacts from the Greatest Hits dataset: a drumstick strike shown at the frame of contact, with example Q values ranging from 223 (glass) to 15 (cloth); per-material medians are given in Section 3.3. (b) Hilbert envelopes plotted against time show that absolute duration is governed by the damping rate $\beta \colon$ the metal bowl $( \beta = \mathrm { \hat { 1 } 4 ~ s ^ { \hat { - } 1 } } )$ rings longest even though glass has the higher $\bar { Q } .$ . (c) Plotted against elapsed oscillation cycles $t f _ { 0 }$ , the ordering follows Q, with cloth and wood falling silent within tens of cycles while glass and metal persist past two hundred. Q therefore measures the quality of resonance rather than its duration. (d) The distillation setup: both encoders stay frozen and only the projection head is trained, with a symmetric InfoNCE objective. No $\bar { Q }$ label enters training, and the ridge probe used for evaluation is fit afterward on held-out videos.

327, glass 412. This ordering is consistent with the extraction capturing physically meaningful variation rather than recording artifacts.

## 4 Physical Structure in Audio-Visual Embeddings

## 4.1 Models Evaluated

We probe four frozen encoders. Two are audio-visual: PE-AV [Vyas et al., 2025], trained contrastively on 100M audio-visual pairs and exposing separate audio, video, and joint embeddings, and CAV-MAE [Gong et al., 2023], a masked autoencoder trained on AudioSet [Gemmeke et al., 2017] with both reconstruction and contrastive objectives. Two are vision-only: CLIP ViT-L/14 [Radford et al., 2021], trained on 400M image–text pairs, and DINOv2 ViT-L/14 [Oquab et al., 2024], trained by self-distillation on curated images without language supervision. The latter pair represents the two dominant recipes for general-purpose visual features, and neither has any acoustic exposure. Video inputs are single frames taken at the annotated moment of impact, each encoded as an image, and audio inputs are 2-second clips centered on the impact.

## 4.2 Probing Protocol

We train ridge regression $( \lambda = 1 . 0 )$ from frozen embeddings to $\log ( Q + 1 )$ and evaluate with Spearman $\rho .$ Pooled correlations carry 95% bootstrap intervals over samples (1,000 resamples). Leave-one-material-out figures are means over the 17 held-out materials, and their intervals resample the materials rather than the samples (10,000 resamples), which is why they are wider.

## 4.3 Results

Table 1 reports the result. The gap between audio $( \rho = 0 . 7 3 )$ and CLIP vision $( \rho = 0 . 2 8 )$ quantifies physical structure that frozen single-frame visual encoders do not linearly expose. Co-training on audio does not by itself place that structure in the visual pathway: PE-AV’s video branch reaches 0.362 and CAV-MAE’s only 0.151, both far below their own audio branches, and the two models differ in architecture, data scale, and objective, none of which this comparison controls. We therefore treat the audio results as motivation for pushing structure into a frozen visual encoder deliberately, rather than as evidence that co-training supplies it for free.

Table 1: Audio embeddings capture the quality factor $^ { Q , }$ while single-frame vision largely does not. Spearman ρ between ridge-probe predictions from frozen embeddings and ground-truth $Q ,$ , on the held-out video split.
<table><tr><td>Model</td><td>Modality</td><td> $\pmb { \rho }$ </td><td>95% CI</td></tr><tr><td>PE-AV</td><td>Audio</td><td>0.727</td><td>[0.707, 0.746]</td></tr><tr><td>CAV-MAE</td><td>Audio</td><td>0.701</td><td>[0.681, 0.721]</td></tr><tr><td>PE-AV</td><td>Joint</td><td>0.696</td><td>[0.676, 0.717]</td></tr><tr><td>CAV-MAE</td><td>Joint</td><td>0.608</td><td>[0.583, 0.633]</td></tr><tr><td>PE-AV</td><td>Video</td><td>0.362</td><td>[0.333, 0.393]</td></tr><tr><td>CLIP</td><td>Video</td><td>0.281</td><td>[0.247, 0.313]</td></tr><tr><td>DINOv2</td><td>Video</td><td>0.271</td><td>[0.242, 0.303]</td></tr><tr><td>CAV-MAE</td><td>Video</td><td>0.151</td><td></td></tr><tr><td></td><td></td><td></td><td>[0.121, 0.181]</td></tr></table>

Table 2: Physical structure generalizes to unseen materials. Mean over the 17 held-out materials, with intervals bootstrapped over materials.
<table><tr><td>Model</td><td>Modality</td><td>LOMO  $\pmb { \rho }$ </td><td>95% CI</td></tr><tr><td>PE-AV</td><td>Audio</td><td>0.600</td><td>[0.528, 0.658]</td></tr><tr><td>CAV-MAE</td><td>Audio</td><td>0.595</td><td>[0.538, 0.645]</td></tr><tr><td>PE-AV</td><td>Joint</td><td>0.572</td><td>[0.505, 0.630]</td></tr><tr><td>CAV-MAE</td><td>Joint</td><td>0.492</td><td>[0.441, 0.541]</td></tr><tr><td>DINOv2</td><td>Video</td><td>0.239</td><td>[0.177, 0.299]</td></tr><tr><td>CLIP</td><td>Video</td><td>0.227</td><td>[0.167, 0.289]</td></tr><tr><td>CAV-MAE</td><td>Video</td><td>0.206</td><td>[0.148, 0.264]</td></tr><tr><td>PE-AV</td><td>Video</td><td>0.197</td><td>[0.144, 0.245]</td></tr></table>

## 5 Generalization Across and Within Materials

## 5.1 Leave-One-Material-Out (LOMO)

High correlation on the full dataset could reflect material-specific memorization. We test generalization: train on 16 materials and evaluate on the held-out 17th.

Audio embeddings predict Q for unseen materials $( \rho \approx 0 . 6 0 ;$ Table 2), and joint embeddings reach 0.49–0.57. This is physical generalization: a probe never trained on the held-out material still ranks its impacts by how much they ring, using only their audio embeddings.

## 5.2 Within Materials, and Which Component of $Q$

Two checks ask whether these correlations are really about physics. The first computes $\rho$ within a single material category, where category identity supplies nothing. The modalities separate sharply: PE-AV audio reaches 0.67 on metal and 0.57 on wood, whereas every visual encoder, including PE-AV’s video branch, is at or below 0.36 on metal and close to zero $( \left| \rho \right| \leq 0 . 1 9 )$ on wood, ceramic, and plastic. CLIP scores 0.06, 0.03, −0.14, and −0.19 on metal, wood, ceramic, and plastic. Baseline vision barely distinguishes physical variation among objects of the same material. Table 7 in Appendix B gives the breakdown.

The second asks which component of $Q$ is responsible, since $Q = \pi f _ { 0 } / \beta$ compounds a spectral term with a damping term, and across our impacts $Q$ is strongly rank-correlated with $f _ { 0 } ( \rho = 0 . 8 7 )$ and moderately anti-correlated with $\beta ( \rho = - 0 . 5 4 )$ . An encoder that merely recovered pitch would therefore score well. Probing the targets separately (Appendix A) shows that the damping term is the one audio captures best: PE-AV audio reaches $\rho = 0 . 7 2 9$ on $\beta$ against 0.700 on $f _ { 0 } ,$ , and the gap widens where category identity stops helping, with $\beta$ holding at 0.686 on unseen materials while $f _ { 0 }$ falls to 0.485. Vision shows the opposite pattern at much lower levels, favoring $f _ { 0 }$ when pooled (0.347 against 0.221), which is consistent with pitch being partly inferable from visible size. The reading we place on $Q$ therefore survives being tested directly, and β is the sharper target.

![](images/69bef3ae2dd7834ab403a8e409b67a22cbb7df709b0fdb975c8394a7d8ff724f.jpg)  
Figure 2: Why pooled correlation is a weak measure of physical understanding. Predicted against true Q on the held-out split $( n = 3 , 5 1 0 )$ , with predictions standardized to the target scale and the diagonal shown for reference. $L e f t { \mathrm { : } }$ a single frame at impact supports only a weak trend. Middle: after distillation, the cloud tightens appreciably. This panel shows the first, and highest, of the three seeds $( \rho = 0 . 4 6 6$ , against the threeseed mean of 0.440 in Table 3). Right: mapping the true material label to its per-category median Q produces just 17 distinct predictions, visible as horizontal bands, and still reaches $\rho = 0 . 4 3 0$ . A model that cannot separate two metal objects at all therefore scores almost as well on this measure as our distilled encoder, which is why the argument rests on Section 6.4 instead.

Table 3: Q prediction from vision on the held-out video split, with and without distillation from the audio teacher.
<table><tr><td>Visual input / model</td><td>ρ</td><td>Interval</td><td>Type</td></tr><tr><td>No audio supervision</td><td></td><td></td><td></td></tr><tr><td>Single frame at impact</td><td>0.281</td><td>[0.247, 0.313]</td><td>95% CI</td></tr><tr><td>Four frames, concatenated</td><td>0.299</td><td>[0.269, 0.332]</td><td>95% CI</td></tr><tr><td>Four frames + deltas</td><td>0.298</td><td>[0.267, 0.329]</td><td>95% CI</td></tr><tr><td>Distilled from the audio teacher</td><td></td><td></td><td></td></tr><tr><td>Four frames + deltas</td><td>0.440</td><td>[0.426, 0.466]</td><td>three-seed range</td></tr><tr><td>Material-label reference (Section 6.5)</td><td>0.430</td><td>[0.400, 0.457]</td><td>95% CI</td></tr><tr><td>Teacher (PE-AV audio)</td><td>0.727</td><td>[0.707, 0.746]</td><td>95% CI</td></tr></table>

## 6 Acoustic-to-Visual Distillation

## 6.1 Setup

The teacher is the frozen PE-AV audio encoder, which predicts Q at $\rho = 0 . 7 3$ . The student is CLIP ViT-L/14, also frozen, with a trainable three-layer MLP head (GELU, LayerNorm, 1024- dimensional output) as the only component that learns. Its input is four frames taken at −0.1, 0, +0.1, and +0.2 s around the annotated impact, each encoded independently and concatenated with their successive differences. Training minimizes a symmetric InfoNCE loss [van den Oord et al., 2018] between the projected visual embedding and the teacher’s audio embedding $( \tau = 0 . 0 7 )$ , with batch size 256 and AdamW [Loshchilov and Hutter, 2019] at a learning rate of $1 \bar { 0 } ^ { - 4 }$

Because the space of plausible configurations is large, we fixed the feature variant and the number of epochs on a validation split drawn from 20% of the training videos (which selected the four frames + deltas input, trained for two epochs) and used the test split for no selection, reporting three seeds. One detail of the data handling carries more weight than it may appear to. We use the dataset’s videolevel split throughout and train the head on training-split impacts only (22,793 of 28,444 impacts, from 780 of 975 videos), because the teacher’s own embeddings predict Q at $\rho = 0 . 7 3 \mathrm { : }$ any impact seen during distillation has effectively had its target leaked, even though no $Q$ label is ever used.

## 6.2 How Much of the Gap Is Temporal?

Before introducing audio at all, we ask whether a single frame is the right visual input. A struck object deforms and recoils, and how quickly that motion subsides is what damping describes.

Widening the window helps, but only modestly on the pooled measure: $\rho$ rises from 0.281 to 0.298 (0.299 without deltas), with overlapping intervals. The effect is clearer under the LOMO protocol

Table 4: True LOMO: the held-out material is excluded from the projection head as well as the probe. Mean over 17 materials and three seeds, with intervals bootstrapped over materials.
<table><tr><td>Model</td><td>LOMO  $\pmb { \rho }$ </td><td>95% CI</td><td>vs. single frame</td></tr><tr><td>Baseline, single frame†</td><td>0.234</td><td>[0.175, 0.289]</td><td></td></tr><tr><td>Baseline, four  $\mathrm { f r a m e s } + \mathrm { d e l t a s } ^ { \dagger }$ </td><td>0.285</td><td>[0.200, 0.372]</td><td>+0.051</td></tr><tr><td>Distilled, four  $\mathrm { f r a m e s } + \mathrm { d e l t a s }$ </td><td>0.338</td><td>[0.249, 0.430]</td><td>+0.104</td></tr></table>

<sup>†</sup>Baselines are recomputed in the impact-window pipeline used for distillation; the cached-feature value for CLIP in Table 2 is 0.227.

Table 5: Controls, as pooled ρ on the held-out video split. Correspondence arms hold the seed fixed, so the matched-teacher row is the first, and highest, of the three seeds (0.466) rather than their mean (0.440).
<table><tr><td>Control Condition</td><td>ρ</td></tr><tr><td>How much is category recognition alone worth? CLIP material classifier (45.6% acc.) → median Q</td><td></td></tr><tr><td>true material label → median Q</td><td>0.279 0.430</td></tr><tr><td>Is the trainable head doing the work? (cached single-frame features†)</td><td></td></tr><tr><td>linear probe on log(Q + 1)</td><td></td></tr><tr><td>three-layer MLP on log(Q + 1), no teacher</td><td>0.300 0.275</td></tr><tr><td>Does the teacher need to correspond to the frame? (four frames + deltas)</td><td></td></tr><tr><td>no distillation</td><td></td></tr><tr><td>shuffled teacher (10.6% material agreement)</td><td>0.298</td></tr><tr><td>positional indexing (13.3% material agreement)</td><td>0.359</td></tr><tr><td>matched teacher</td><td>0.360 0.466</td></tr></table>

<sup>†</sup>The capacity pair is measured on the cached single-frame embeddings rather than the impact-window pipeline; the two rows are comparable with each other, and their values (0.275, 0.300) bracket the 0.281 single-frame figure.

of Section 6.4, where the same change is worth +0.051 (0.234 to 0.285). We therefore treat the temporal contribution as modest, while noting that it costs no acoustic supervision whatsoever.

## 6.3 Distillation Results

Distillation raises vision-only Q prediction to 0.440, from 0.298 with the same four-frame input and no teacher. This is marginally above the 0.430 obtainable from perfect material-category knowledge and still far below the audio teacher. Because the pooled measure is dominated in practice by category information, exceeding that reference by 0.01, which lies inside the three-seed range, is not conclusive (Figure 2), so we turn to a measure under which category identity earns nothing.

## 6.4 Leave-One-Material-Out, with the Head Held Out

LOMO is the discriminating test, because correlations are computed within a held-out material: a model that outputs a category’s median Q predicts a constant there and is scored zero, since Spearman $\rho$ is undefined for constant predictions. Two conditions must both hold for the test to mean anything. The material must be excluded from the probe, and it must also be excluded from the distillation head—otherwise the head has already learned that material’s appearance-to-audio mapping and the probe merely reads it back. We refer to this stricter protocol as true LOMO.

Table 4 reports the outcome. Distillation improves LOMO by +0.104 over the single-frame baseline, with 15 of 17 materials improving (Wilcoxon $p = 0 . 0 0 2 )$ , and by +0.053 over the four-frame baseline, with 14 of 17 improving $( p = 0 . 0 0 1 )$ . The gains are uneven: the five largest (single-frame baseline → distilled) are on granular and natural surfaces, namely gravel 0.364 → 0.708, grass 0.377 → 0.610, dirt 0.285 → 0.605, leaf $0 . 2 9 0  0 . 5 4 1$ , and rock $0 . 2 4 7  0 . 4 6 9$ , which together account for about three-quarters of the total gain.

## 6.5 Controls

Table 5 collects the three controls, which together bound what this result can mean.

The first control asks how much of any $Q$ correlation is simply material recognition. Mapping the true material to its per-category median $Q$ gives $\rho = 0 . 4 3 0 \ [ 0 . 4 0 0 , 0 . 4 5 7 ]$ , and a material classifier trained on CLIP features, at 45.6% accuracy, mapped the same way gives $\rho = 0 . 2 7 9 [ 0 . 2 4 6 , 0 . 3 1 1 ]$ Any pooled correlation near 0.28 is therefore indistinguishable from category recognition, which is precisely why we rest the argument on LOMO rather than on the pooled figure.

The second asks whether the trainable head is doing the work rather than the teacher. An identically shaped head given direct supervision on log(Q+1), with no audio involved, reaches only $\rho = 0 . 2 7 5$ [0.224, 0.293], below the 0.300 [0.267, 0.330] reached by a linear probe on the same features. Additional parameters buy nothing here: with roughly 14,000 noisy scalar targets, a regression head has little to learn from, whereas the teacher supplies a dense 1024-dimensional target whose geometry already encodes damping.

The third asks whether any structured target would do. Holding the seed fixed, distilling from correctly matched audio reaches $\rho = 0 . 4 6 6$ (the first, and highest, of the three seeds, whose mean is the 0.440 of Table 3) against 0.298 with no distillation. Distilling from a fully shuffled teacher (10.6% material agreement) still reaches 0.359, and positional indexing, used by an earlier version of our pipeline (13.3% material agreement), reaches 0.360. Roughly a third of the raw gain therefore survives destroying the correspondence entirely, while the larger part, +0.107, requires the teacher to correspond to the frame it is paired with. Reported gains from cross-modal distillation should be read against a shuffled-teacher control for this reason.

## 7 Discussion

Three findings survive the controls. First, damping structure is present in audio and largely absent from single-frame vision, and separating the components of Q shows that audio carries the damping term $\beta$ better than pitch, which is what licenses reading the transferred structure as physical. Second, a frozen visual encoder can acquire part of that structure from a frozen audio teacher without any $Q$ label entering training, and the gain holds on materials withheld from the projection head as well as the probe (0.234 to 0.338, 15 of 17 materials), where a category-level strategy earns nothing. Third, roughly half of the improvement needs no audio at all and follows from four frames instead of one, so part of what looks like a limit of vision is a limit of single-frame inputs.

The effect is modest in absolute terms. Most of the teacher’s signal does not cross, and on the pooled measure the distilled encoder barely exceeds what perfect category knowledge would supply. That comparison is itself the methodological result we would most like carried forward: category identity alone reaches $\rho = 0 . 4 3$ on this task, so a pooled number near that value says little about physical understanding. Establishing physical transfer needs an evaluation under which category recognition earns nothing, namely true LOMO, together with a shuffled-teacher control (Section 6.5).

The residual is a concrete target rather than a closed question. Damping depends on internal structure a camera often cannot observe, and the natural next step is $\beta$ itself, the component audio predicts best, nearly independent of $f _ { 0 } ,$ and the one on which the distilled student already improves most.

## 8 Limitations

Four caveats bound these results. First, evaluation is confined to Greatest Hits (977 videos, 17 materials) and to one visual domain, so whether the projection head learns a general appearance-todamping mapping or associations tied to controlled drumstick footage is unresolved. Second, the $R ^ { 2 } \bar { \geq } 0 . 8$ filter keeps 63% of annotated impacts at strongly material-dependent rates (Section 3.2), and the five largest per-material gains of Section 6.4 fall on materials retaining a third to just over a half of their impacts, so those correlations cover a selected minority of each category. The discarded impacts cannot be assessed with $Q$ at all, since there the target is undefined. Third, the four-frame model is evaluated on $Q$ alone (the $f _ { 0 } / \beta$ probe in Appendix A uses a single-frame student), so although probing shows that audio carries $\beta$ better than $f _ { 0 }$ (Section 5.2), we do not establish transfer of physical structure in general, and properties that impact sound encodes weakly or not at all, such as friction, may behave differently. Finally, we measure transferred structure by probing accuracy alone, so whether better $Q$ decodability yields better physical decisions remains to be shown on a task whose outcome is not a function of ${ \dot { Q } } .$

## References

Relja Arandjelovic and Andrew Zisserman. Look, listen and learn. In´ Proceedings of the IEEE International Conference on Computer Vision (ICCV), pages 609–617, 2017.

Yusuf Aytar, Carl Vondrick, and Antonio Torralba. Soundnet: Learning sound representations from unlabeled video. In Advances in Neural Information Processing Systems (NeurIPS), volume 29, 2016.

Daniel M Bear, Elias Wang, Damian Mrowca, Felix J Binder, Hsiao-Yu Fish Tung, RT Pramod, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, et al. Physion: Evaluating physical prediction from vision in humans and machines. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2021.

Wei Chow, Jiageng Mao, Boyi Li, Daniel Seita, Vitor Guizilini, and Yue Wang. PhysBench: Benchmarking and enhancing vision-language models for physical world understanding. In International Conference on Learning Representations (ICLR), 2025.

Samuel Clarke, Ruohan Gao, Mason Wang, Mark Rau, Julia Xu, Jui-Hsien Wang, Doug L James, and Jiajun Wu. RealImpact: A dataset of impact sound fields for real objects. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 1516– 1525, 2023.

Ruohan Gao, Yiming Dou, Hao Li, Tanmay Agarwal, Jeannette Bohg, Yunzhu Li, Li Fei-Fei, and Jiajun Wu. The ObjectFolder benchmark: Multisensory learning with neural and real objects. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 17276–17286, 2023.

Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter. Audio Set: An ontology and human-labeled dataset for audio events. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.

Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. ImageBind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180– 15190, 2023.

Yuan Gong, Andrew Rouditchenko, Alexander H Liu, David Harwath, Leonid Karlinsky, Hilde Kuehne, and James Glass. Contrastive audio-visual masked autoencoder. In Proceedings of the International Conference on Learning Representations (ICLR), 2023.

Saurabh Gupta, Judy Hoffman, and Jitendra Malik. Cross modal distillation for supervision transfer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2827–2836, 2016.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Po-Yao Huang, Vasu Sharma, Hu Xu, Chaitanya Ryali, Yanghao Li, Shang-Wen Li, Gargi Ghosh, Jitendra Malik, Christoph Feichtenhofer, et al. MAViL: Masked audio-video learners. Advances in Neural Information Processing Systems, 36:20371–20393, 2023.

Jiaxun Liu and Boyuan Chen. SonicSense: Object perception from in-hand acoustic vibration. In Conference on Robot Learning (CoRL), 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

Andrew Owens and Alexei A Efros. Audio-visual scene analysis with self-supervised multisensory features. In Proceedings of the European Conference on Computer Vision (ECCV), pages 631– 648, 2018.

Andrew Owens, Phillip Isola, Josh McDermott, Antonio Torralba, Edward H Adelson, and William T Freeman. Visually indicated sounds. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2405–2413, 2016a.

Andrew Owens, Jiajun Wu, Josh H McDermott, William T Freeman, and Antonio Torralba. Ambient sound provides supervision for visual learning. In Proceedings of the European Conference on Computer Vision (ECCV), pages 801–816, 2016b.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), 2021.

Ronan Riochet, Mario Ynocente Castro, Mathieu Bernard, Adam Lerer, Rob Fergus, Véronique Izard, and Emmanuel Dupoux. IntPhys 2019: A benchmark for visual intuitive physics understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5016–5025, 2021.

Abitha Thankaraj and Lerrel Pinto. That sounds right: Auditory self-supervision for dynamic robot manipulation. In Conference on Robot Learning, pages 1036–1049. PMLR, 2023.

Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel Bear, Chuang Gan, Josh Tenenbaum, Dan Yamins, Judith Fan, and Kevin Smith. Physion++: Evaluating physical scene understanding that requires online inference of different physical properties. Advances in Neural Information Processing Systems, 36:67048–67068, 2023.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Apoorv Vyas, Heng-Jui Chang, Cheng-Fu Yang, Po-Yao Huang, Luya Gao, Julius Richter, Sanyuan Chen, Matt Le, Piotr Dollár, Christoph Feichtenhofer, et al. Pushing the frontier of audiovisual perception with large-scale multimodal correspondence learning. arXiv preprint arXiv:2512.19687, 2025.

Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. CLEVRER: Collision events for video representation and reasoning. In International Conference on Learning Representations (ICLR), 2020.

## A Separate Probes for $f _ { 0 }$ and $\beta$

Table 6 probes the components of $Q$ separately, under the protocol of Section 4.2: ridge on frozen embeddings, Spearman $\rho$ on the held-out video split, and LOMO trained on 16 materials and evaluated within the held-out 17th. Targets are log-transformed as elsewhere. Across the 17,958 impacts, Q is strongly rank-correlated with $f _ { 0 } \left( \rho = 0 . 8 7 \right)$ and moderately anti-correlated with $\beta \left( \rho = - 0 . 5 4 \right)$ while the two components are close to independent of each other $( \rho = - 0 . 1 1 )$

Two patterns are worth drawing out. First, for both audio encoders and the PE-AV joint embedding, $\beta$ is predicted better than $f _ { 0 } ,$ , and the margin grows under LOMO, which is where category identity stops carrying the prediction. Second, the visual encoders order the targets the other way when pooled, favoring $f _ { 0 } ,$ which is consistent with pitch being partly inferable from visible object size, and that ordering reverses on unseen materials. Under true LOMO, the distilled student improves on both targets and improves most on $\beta \ ( 0 . 2 5 7  0 . 3 7 5 )$ , so what transfers is preferentially the damping term rather than the spectral one.

## B Within-Material Q Prediction

Table 7 gives the per-category breakdown summarized in Section 5.2. Correlations are computed within each material, so category identity carries no information.

Table 6: Probing $f _ { 0 }$ and $\beta$ separately. Audio embeddings predict the damping term $\beta$ better than the spectral term $f _ { 0 } ,$ and the margin widens on unseen materials. Bold marks the better of $f _ { 0 }$ and $\beta$ within each row and setting. The distilled row’s LOMO columns use true LOMO, with the held-out material excluded from both the projection head and the probe, and comes from a single-frame student (the frame at impact), trained separately from the four-frame model of Tables 3–4, so it is compared like-for-like with the single-frame CLIP row.
<table><tr><td rowspan="2">Model</td><td colspan="2">Pooled  $\pmb { \rho }$ </td><td colspan="2">LOMO  $\pmb { \rho }$ </td></tr><tr><td> $f _ { 0 }$ </td><td> $\beta$ </td><td> $f _ { 0 }$ </td><td> $\beta$ </td></tr><tr><td>PE-AV (audio)</td><td>0.700</td><td>0.729</td><td>0.485</td><td>0.686</td></tr><tr><td>CAV-MAE (audio)</td><td>0.667</td><td>0.732</td><td>0.423</td><td>0.719</td></tr><tr><td>PE-AV (joint)</td><td>0.671</td><td>0.682</td><td>0.439</td><td>0.628</td></tr><tr><td>CLIP</td><td>0.347</td><td>0.221</td><td>0.137</td><td>0.257</td></tr><tr><td>DINOv2</td><td>0.353</td><td>0.210</td><td>0.152</td><td>0.244</td></tr><tr><td>PE-AV (video)</td><td>0.396</td><td>0.337</td><td>0.134</td><td>0.207</td></tr><tr><td>CLIP distilled</td><td>0.345</td><td>0.311</td><td>0.177</td><td>0.375</td></tr></table>

Table 7: Within-material Q prediction $( \rho ) ,$ shown for four representative rigid materials.
<table><tr><td>Model</td><td>Metal</td><td>Wood</td><td>Ceramic</td><td>Plastic</td></tr><tr><td>PE-AV (audio)</td><td>0.67</td><td>0.57</td><td>0.31</td><td>0.45</td></tr><tr><td>CAV-MAE (audio)</td><td>0.65</td><td>0.45</td><td>0.12</td><td>0.32</td></tr><tr><td>PE-AV (joint)</td><td>0.64</td><td>0.41</td><td>0.02</td><td>0.34</td></tr><tr><td>CAV-MAE (joint)</td><td>0.43</td><td>0.25</td><td>-0.10</td><td>0.18</td></tr><tr><td>PE-AV (video)</td><td>0.36</td><td>0.07</td><td>-0.08</td><td>0.00</td></tr><tr><td>DINOv2</td><td>0.30</td><td>0.09</td><td>0.09</td><td>-0.03</td></tr><tr><td>CLIP†</td><td>0.06</td><td>0.03</td><td>-0.14</td><td>-0.19</td></tr><tr><td>CAV-MAE (video)</td><td>-0.21</td><td>-0.01</td><td>0.06</td><td>-0.14</td></tr></table>

<sup>†</sup>CLIP is measured on the impact-window pipeline with verified impact-to-embedding alignment, whereas the other rows come from the cached pipeline.