# Motion-Based Tokenization for Cross-Dataset Egocentric Gaze Modeling

Virmarie Maquiling<sup>1,2</sup> Zhuojiang Cai<sup>1,2</sup> Enkelejda Kasneci<sup>1,2</sup>

<sup>1</sup>Human-Centered Technologies for Learning, Technical University of Munich (TUM), Munich, Germany <sup>2</sup>Munich Center for Machine Learning (MCML), Munich, Germany

virmarie.maquiling@tum.de

## Abstract

Gaze is increasingly used as an input signal for vision and multimodal models, yet no consensus exists on how to represent it across datasets. Raw traces preserve detail but are noisy and device-dependent, while coarse event labels are easy to model but can discard local motion structure. We formulate event-aligned, fixed-horizon angular displacement as an interpretable, event-conditioned motion vocabulary and compare it with event-only, spatial, absolute-angle, learned vector-quantized, and continuous representations. To assess transfer alongside target predictability and token collapse, our evaluation combines next-token prediction with target-domain regret, low-order target references, paired bootstrap, order sensitivity, motif overlap, and frozen structural probes. In an eventaligned headset benchmark, angular-motion tokens have lower target-domain regret thanfrozen-codebook VQ tokens in one transfer direction, while the reverse direction is inconclusive. The probes reveal complementary representation properties, and event-only tokens show that low perplexity can retain little motion information. On a third egocentric dataset, a matched comparison of I-VT, native, and frame-span interfaces shows that event construction materially changes transfer: native events have the lowest regret into EGTEA, while frame-span events have zero motif overlap and fail severely as a source. Motion-based tokenization therefore provides a compact representation for event-aligned egocentric gaze streams, while the evaluation identifies how target predictability and event construction shape cross-dataset conclusions.

## 1. Introduction

For gaze to become a useful input signal in vision models, its representation must be compact enough to model, stable enough to compare across datasets, and structured enough to preserve temporal dependencies. Raw gaze traces retain detail but are noisy, device-dependent, and difficult to align across studies. Coarse event labels, such as fixations and saccades, are easier to model but can discard local motion structure, while spatial tokenizations can become brittle under domain shift. The representation therefore determines which behavioral regularities remain available across datasets. Gaze is highly sensitive to task and behavioral context. Classical work showed that eye movements can change substantially even under a fixed visual stimulus [25], while recent works increasingly use gaze for diverse use cases such as mind wandering, attention prediction, action prediction, and multimodal supervision [4, 11, 17, 28, 29]. Yet gaze is still often represented either as a low-level continuous trace or through discretizations tied to a particular task or dataset. We ask whether local gaze motion can instead be discretized into tokens that remain predictive and structurally meaningful beyond the dataset on which a model is trained.

While gaze shifts have commonly been characterized in eye movement analysis by their amplitude and direction, their use, particularly in the form of angular displacement, as an event-conditioned discrete vocabulary for crossdataset sequence modeling remains underexplored. We therefore formulate event-aligned, fixed-horizon angular change as an event-conditioned discrete vocabulary that reduces dependence on absolute gaze position under a shared event and coordinate interface. This does not make the representation generally invariant. Angular differencing does not remove arbitrary frame rotations, nonlinear calibration errors, temporal resampling, changes in event boundaries, or device-dependent noise.

We compare angular-motion tokens with event-only, spatial, absolute-angle, learned vector-quantized, and continuous representations on three egocentric eye-tracking datasets with different behavioral regimes. AEA [20] and RITW [24] form the main cross-dataset transfer benchmark. Because both were recorded with Meta’s Project Aria headset, they provide a conservative setting in which sensor geometry, coordinate conventions, and preprocessing assumptions are relatively aligned. EGTEA [18], recorded with a different tracker and exported through BeGaze, probes the boundary of this setting by testing whether the same motion vocabulary remains useful when the device and eventconstruction interface change.

Cross-dataset conclusions also depend on how transfer is evaluated. A representation can achieve low out-of-domain perplexity because the target stream is easier to predict, while a small or nearly deterministic vocabulary can appear stable despite retaining little motion information. We therefore combine degradation with target-domain regret and low-order target references, paired bootstrap, order sensitivity, motif overlap, and frozen structural probes. We further stress-test the conclusions across tokenizer configurations, temporal scales, VQ codebook policies, and alternative event constructions. This evaluation jointly examines predictive transfer, target predictability, and structural retention.

The results are quite direction-specific. Angularmotion tokens have lower target-domain regret than frozen-codebook VQ tokens for AEA→RITW, while RITW→AEA is inconclusive under paired bootstrap. Angular-motion tokens nevertheless retain stronger order dependence and motif overlap, while the controlled probes show complementary representation properties. On EGTEA, a matched three-interface comparison shows that event construction materially changes transfer: native events have the lowest regret into EGTEA, degree-aware I-VT retains substantial motif overlap, and frame-span events have zero motif overlap and severe EGTEA-as-source transfer. These findings show that motion-token transfer depends not only on the vocabulary, but also on the temporal and event interfaces through which gaze is represented.

Our contributions are fourfold. First, we formulate event-aligned, fixed-horizon angular displacement as an interpretable, event-conditioned motion vocabulary and compare it with discrete, learned, and continuous alternatives under cross-dataset transfer. Second, we provide an evaluation protocol that contextualizes transfer using target references and tests whether temporal and geometric structure remain available. Third, we analyze tokenizer sensitivity, temporal granularity, codebook policy, and representationspecific failure modes. Fourth, we compare degree-aware I-VT, vendor-native, and frame-span constructions on a third egocentric dataset under matched training and evaluation controls, showing that event construction is part of the representation rather than neutral preprocessing.

## 2. Related work

Gaze representation. The choice of gaze representation is not neutral, as different formulations preserve different aspects of behavior while discarding others. Classical saccade analysis characterizes movements through angular amplitude, duration, and velocity [3], while Scan-Match [8] converts fixations into spatial-temporal letter sequences, and MultiMatch [9] compares scanpaths across shape, position, direction, length, and duration. Other sequence-based methods represent gaze through latent-state transitions [6, 7], recurrence patterns [1], or short subsequence frequencies [15]. These methods differ in what structure they make measurable, reinforcing that representation choice is part of the modeling problem.

Discrete gaze tokenization. Discrete scanpath representations are common in comparison and prediction: they make alignment, sequence modeling, and motif analysis tractable, but trade off fidelity for stability [2, 16]. Recent work explicitly compares tokenization strategies for gaze forecasting and generation with language-model-style architectures [22]. Our focus takes a complementary evaluation angle. Beyond asking whether gaze should be continuous or discrete, we ask which abstraction preserves reusable sequential structure under dataset shift and how conclusions change with target-domain predictability, codebook construction, and event definition. To this end, we treat angular-delta tokenization as a representation family and test its robustness to grid size, stride, event conditioning, zero-motion handling, and out-of-range handling.

Domain shift and transfer. Domain shift is well known in gaze estimation, where models degrade across datasets because capture conditions, subjects, gaze ranges, and annotation procedures change [13, 26, 27]. Higher-level gaze models also increasingly use structured gaze or personspecific tokens to improve generalization [12, 19, 23]. Most work measures robustness through end-task accuracy. Instead, we ask whether the representation itself remains useful under shift: compact enough to be learned efficiently, structured enough to support prediction, and stable enough to preserve local sequential structure across datasets.

## 3. Method and Experimental Setup

We study cross-domain gaze sequence modeling to determine which representations remain useful under dataset shift. Given a source dataset, we tokenize gaze events, train a next-token model on the source, and evaluate it both indomain and on held-out target-domain streams. Here, transfer refers to predictive and structural retention under a specified source–target protocol. It is not assumed to hold across incompatible event, coordinate, temporal, or task interfaces.

## 3.1. Datasets

We use three egocentric eye-tracking datasets with distinct behavioral regimes: AEA (general everyday activities) [20], RITW (Reading in the Wild–Columbus dataset) [24], and EGTEA (cooking activities) [18]. AEA and RITW were both recorded with Meta’s Project Aria Gen 1 headset [10], while EGTEA was recorded with an SMI wearable eye tracker [18]. In the gaze exports used in our pipeline, AEA and RITW differ substantially in effective sampling density. EGTEA differs in both sensor and export format, as it is provided through BeGaze exports with frame-aligned gaze samples and native fixation/saccade annotations. Consequently, AEA and RITW are used as the main cross-dataset transfer pair and EGTEA as an event-interface boundary check.

## 3.2. Representations and models

All discrete representations are derived from event streams. For AEA and RITW, we obtain events using a simplified adaptive velocity-threshold identification (I-VT) procedure inspired by Nystrom and Holmqvist¨ [21], with a median absolute deviation (MAD)-based per-sequence velocity threshold. For EGTEA, we compare three event interfaces. Degree-aware I-VT constructs fixation/saccade events after applying 360<sup>◦</sup> angular unwrapping to the projection-based pseudo-yaw/pitch values. The native interface uses the BeGaze fixation/saccade annotations and averages valid samples within each interval. The frame-span interface emits one fixation-typed event per retained video frame within contiguous valid spans. All three interfaces use the same AD tokenizer. Full construction and health statistics are provided in the supplement.

Our primary representation is ang delta (AD), a deterministic motion tokenizer that quantizes eventconditioned (∆yaw, ∆pitch) on a fixed 2D grid, with explicit zero-motion tokens and configurable out-of-range handling. Rather than differencing adjacent event centers, the default AD interpolates yaw/pitch 80 ms after each valid event, subject to the stride tolerance, and quantizes the resulting displacement. If no valid target exists, it emits the event-specific zero token. We compare AD with anchor ang (AA), discretized absolute yaw/pitch; ang state and motion (ASM), interleaved absolute state and local motion; so3 delta (SD), the SO(3) log-map between consecutive gaze rays; event only (EVT), an event-label baseline; spatial grid+event (SGE), event type combined with a fixed yaw/pitch cell; and vq delta (VQD), which maps normalized (∆yaw, ∆pitch) to the nearest frozen-codebook centroid. For event-conditioned tokenizers, the event label and representation bin jointly index one token. The Transformer uses a single embedding lookup without a separate eventtype embedding. For VQD, we evaluate K = 128 codebooks over three seeds under source-specific, AEA-reuse, and pooled AEA+RITW fitting. All codebooks use training deltas only. A fixed-seed K = 256 result provides a secondary size diagnostic. For the controlled AD–VQD regret comparison, the source-fitted K = 128 codebook remains frozen during source-model training, target-reference training, and target evaluation, ensuring a fixed discrete alpha-

bet.

All discrete representations share the same next-token Transformer architecture, splits, context length, minimum sequence-length filter, optimizer family, and training budget. Only the AD temporal-scale analysis in Section 4.4 varies stride and context length. The supplement details each tokenizer. We also include a continuous autoregressive baseline (CONT), which predicts normalized continuous ∆yaw and ∆pitch using a Transformer trained with Gaussian negative log-likelihood. Because it belongs to a different metric family, CONT is reported separately. Model and training details appear in the supplement.

## 3.3. Evaluation protocol

For discrete models, we accumulate token negative loglikelihood in nats. We report next-token perplexity and convert target cross-entropy to bits/token for regret:

$$
\begin{array} { l } { { \displaystyle { \cal H } _ { T } ^ { \mathrm { n a t } } ( q ) = - \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \ln q ( x _ { t } \mid x _ { < t } ) , } } \\ { { \displaystyle ~ { \cal H } _ { T } ( q ) = \frac { { \cal H } _ { T } ^ { \mathrm { n a t } } ( q ) } { \ln 2 } , ~ \mathrm { P P L } ( q ) = \exp \left( { \cal H } _ { T } ^ { \mathrm { n a t } } ( q ) \right) . } } \end{array}\tag{1}
$$

$$
\mathrm { D e g r a d a t i o n } = \frac { \mathrm { M e t r i c } _ { \mathrm { O O D } } - \mathrm { M e t r i c } _ { \mathrm { I N D } } } { \mathrm { M e t r i c } _ { \mathrm { I N D } } } .\tag{2}
$$

Lower perplexity is better, while smaller degradation indicates greater stability under shift. Because degradation neither controls for target predictability nor provides statistical separation, and may be negative when the target stream is easier, we interpret it alongside regret and structural diagnostics. Likewise, alphabets and token rates differ—ASM emits two tokens per event— so perplexity and bits/token are representation-specific, and equal contexts may cover different event histories. They diagnose transfer, not motion fidelity.

Target-domain regret partially controls for target predictability by comparing source- and target-trained models. Let $H _ { T }$ be cross-entropy on the held-out target test split, q<sub>S</sub> a source-trained model, $q _ { T }$ the same architecture trained on the target training split, and $q _ { B }$ a target-trained unigram or first-order Markov baseline. All are evaluated on the same target test split.

$$
\begin{array} { r } { R _ { \mathrm { r a w } } ( S \to T ) = H _ { T } ( q _ { S } ) - H _ { T } ( q _ { T } ) , } \\ { R _ { \mathrm { n o r m } } ^ { ( B ) } ( S \to T ) = \frac { H _ { T } ( q _ { S } ) - H _ { T } ( q _ { T } ) } { H _ { T } ( q _ { B } ) - H _ { T } ( q _ { T } ) } . } \end{array}\tag{3}
$$

Normalized regret above one means that source-trained transfer remains worse than baseline B relative to the targettrained reference. A non-positive denominator makes normalized regret undefined and is reported as NA. For AD– VQD comparisons, we derive matched differences from sequence-level losses weighted by valid prediction counts and obtain 95% confidence intervals using a hierarchical bootstrap over seeds and target sequences.

To verify that models capture sequence structure beyond merely token frequencies, we use order-manipulation controls based on perplexity: RR (train real, test real), RS (train real, test shuffled), and SR (train shuffled, test real). Using these controls, we define:

$$
{ \begin{array} { r } { { \mathrm { O r d e r ~ S e n s i t i v i t y } } = { \frac { \mathrm { R S - R R } } { \mathrm { R R } } } , } \\ { { \mathrm { L e a r n i n g ~ G a p } } = { \frac { \mathrm { S R - R R } } { \mathrm { R R } } } . } \end{array} }
$$

Order sensitivity measures the effect of disrupting test order, while learning gap measures the cost of training on shuffled sequences. We additionally measure local crossdomain compatibility using bigram and trigram Jaccard overlap and the target n-gram mass unseen in the source. Counting and top-K details are given in the supplement.

As structural diagnostics, we train linear probes on frozen, mean-pooled sequence-model states to predict motion-state, motion-quadrant, and direction-reversal decodability. These assess local motion-state, motionquadrant, and reversal information. The supplement provides full probe details and a secondary within-dataset RITW reading-state probe.

We analyze AD temporal scale by sweeping eventpair stride (delta stride ms) and context length (context len) in both transfer directions. Stride controls motion per token, while context length controls the available token history. We report OOD perplexity and degradation and examine token-distribution statistics and Kneser– Ney-smoothed n-gram predictability [5, 14] to distinguish compression from aggregation effects.

We also test AD sensitivity to angular-grid resolution, event-pair stride, and event-type conditioning over multiple seeds. Each variant uses the full AEA↔RITW protocol and reports IND/OOD perplexity, degradation, order sensitivity, entropy, top-10 concentration, and vocabulary usage. Fixed-seed checks cover zero-motion and out-of-range handling.

For EGTEA, we compare degree-aware I-VT, vendornative, and frame-span interfaces. All conditions use fixed context 8, minimum sequence length 20, three seeds, identical session-level 70/15/15 splits, uniform session sampling, and matched per-session window caps. Transfer degradation, target-domain regret, motif overlap, and sourcemismatch statistics are evaluated on the same 86 sessions.

## 3.4. Reproducibility

We repeat the main AEA↔RITW comparison, tokenizersensitivity settings, continuous baseline, and structuralprobe diagnostics over three seeds and report mean and standard deviation. Deterministic tokenizers have fixed token streams and varying model seeds. VQD uses source-specific K = 128 codebook seeds, and CONT is trained separately per seed. The matched EGTEA event-interface comparison uses three seeds and population standard deviations. The SO(3) ablations remain fixed-seed seed-0 diagnostics. Code and settings are available at https://anonymous.4open.science/r/ motionTokenizer\_egocentric-BE1E/.

## 4. Results

## 4.1. Cross-domain transfer

Our main discrete benchmark is AEA↔RITW (see Figure 1 for a visualization of the transfer results). Excluding the geometry-free EVT control, AD has the lowest observed pooled mean degradation among the hand-designed tokenizers. It obtains degradation 0.05±0.07, lower than ASM (0.15±0.12), SGE (0.19±0.22), AA (0.21±0.18), and SD $( 1 . 3 0 \pm 0 . 6 0 ) $ . Direction-specific degradation is 0.01 ± 0.08 for AEA→RITW and 0.09 ± 0.04 for RITW→AEA. Because degradation does not control for target-domain predictability, we next compare AD and VQD using targetdomain regret.

Because Table 1 uses a padded single-target context-16 protocol, its cross-entropies are not comparable with Figure 1A and support only within-protocol regret comparisons. In AEA→RITW, AD has lower raw target-domain regret than VQD (0.634 vs. 1.620 bits/token), and VQD remains worse than both target-domain low-order baselines after normalization. In RITW→AEA, raw regrets are close (0.503 for AD and 0.528 for VQD), and VQD normalized regret is undefined because the low-order baseline denominator is non-positive.

The paired AD–VQD raw-regret difference is −0.964 bits/token for AEA→RITW, with 95% CI [−1.183, −0.756], confirming lower AD regret. For RITW→AEA, the difference is 0.066, with 95% CI [−0.417, 0.616], making this direction inconclusive. Because these estimates use matched sequence-level losses, they need not equal differences between the seed-level means in Table 1. Full bootstrap results are reported in the supplement.

EVT shows why perplexity is not enough on its own. Its near-perfect AEA→RITW transfer follows from a tiny, highly predictable vocabulary, but it contains no geometric or motion information. In the opposite direction, the metric becomes degenerate. Its high order sensitivity reflects an almost deterministic event sequence that is fragile when order is disrupted. EVT is thus used only as a lowinformation sanity check. This illustrates a general failure mode for token-level transfer metrics: a representation can appear robust because it has collapsed the behavioral signal into a small, nearly deterministic alphabet. For this reason, we interpret low degradation only together with structural diagnostics that test whether geometry and temporal order remain available.

Table 1. Target-domain regret for AD and VQD under the controlled context-16 $\mathrm { A E A } {  } \mathrm { R I T W }$ comparison. $H _ { T } ( q _ { S } )$ and $H _ { T } ( q _ { T } )$ are target-test cross-entropies in bits/token for source- and target-trained models. Lower raw regret is better. Normalized values above one indicate worse performance than the corresponding target low-order baseline. Values are means over seeds. VQD normalization fo $\mathrm { R I T W } {  } \mathrm { A E A }$ is NA because the denominator is non-positive.
<table><tr><td>Transfer</td><td>Rep.</td><td> $H _ { T } ( q _ { S } )$ </td><td> $H _ { T } ( q _ { T } )$ </td><td>Raw regret ↓</td><td>Unigram norm. ↓</td><td>Markov norm. ↓</td></tr><tr><td> $\mathrm { A E A \to R I T W }$ </td><td>AD</td><td>2.924</td><td>2.290</td><td>0.634</td><td>0.450</td><td>1.958</td></tr><tr><td>AEA→RITW</td><td>VQD</td><td>5.291</td><td>3.672</td><td>1.620</td><td>2.516</td><td>3.889</td></tr><tr><td>RITW→AEA</td><td>AD</td><td>3.529</td><td>3.026</td><td>0.503</td><td>0.373</td><td>0.265</td></tr><tr><td> $\mathrm { R I T W } {  } \mathrm { A E A }$ </td><td>VQD</td><td>7.342</td><td>6.814</td><td>0.528</td><td>NA</td><td>NA</td></tr></table>

SD shows that preserving more geometric detail is not automatically beneficial. The default full angle-axis $\mathrm { S O ( 3 ) }$ tokenizer transfers poorly in both directions and has almost no shared local motif structure, with bigram/trigram overlap of only 0.054/0.002. This failure is not mainly due to outside buckets or missing individual tokens, given that outside clipping changes little and unigram support overlap remains moderate. Instead, the ablation suggests axis-conditioned fragmentation. Collapsing SD to an angle-only variant recovers much stronger transfer and motif overlap, raising bigram/trigram overlap to 0.683/0.418. SD thus serves as a useful failure case, showing how fine-grained 3D rotation tokens can over-partition local gaze motion, making transition structure less reusable across datasets. Further SO(3) ablations are reported in the supplement.

## 4.2. Representation diagnostics

Order sensitivity and learning gap. Figures 1C–D show that AD depends meaningfully on real sequence order. Across three seeds, AD has pooled order sensitivity $0 . 4 0 \pm$ 0.10, lower than ASM (0.47±0.11) but substantially higher than $\mathrm { A A } \left( 0 . 1 4 \pm 0 . 0 4 \right)$ , SGE $( 0 . 1 5 \pm 0 . 0 4 )$ , and SD $( 0 . 1 7 \pm$ 0.12). This pattern suggests that both AD and ASM capture sequential structure.

The learning-gap control gives a compatible picture. AD has a moderate pooled learning gap $( 0 . 0 5 \pm 0 . 0 2 ) $ , while ASM is higher $( 0 . 1 2 \pm 0 . 0 5 )$ . AA, SGE, and SD remain lower or noisier. Thus, AD combines relatively low degradation with substantial order sensitivity. Motif overlap further separates the representations: AD has much higher bigram/trigram Jaccard overlap (0.493/0.449) than ASM (0.173/0.146), AA (0.031/0.003), SGE (0.042/0.003), or SD (0.034/0.005). This supports the interpretation that AD preserves reusable local transition structure under the shared AEA/RITW event and coordinate interface.

Probe-based diagnostics. Figure 1E summarizes the matched context-16 bidirectional frozen-probe rerun using the source-specific K = 128 VQD policy. Reported means and standard deviations pool both transfer directions over three seeds, giving six direction-by-seed values. The probes show complementary strengths. CONT has the highest mean motion-quadrant decodability $( 0 . 6 2 5 \pm 0 . 1 0 1 )$ while AD is highest among the discrete tokenizers (0.481 ± 0.044), compared with EVT (0.224±0.012), SGE (0.270± 0.028), and VQD $( 0 . 2 6 7 \pm 0 . 0 2 5 )$ . Direction-reversal decodability is less separated: CONT reaches $0 . 5 3 5 \pm 0 . 0 4 4$ while AD, VQD, SGE, and EVT reach $0 . 4 9 7 \pm 0 . 0 4 4 .$ $0 . 4 9 5 \pm 0 . 0 2 6 , 0 . 4 7 3 \pm 0 . 0 3 1$ , and $0 . 4 1 3 \pm 0 . 0 2 9$ , respectively.

Motion detection instead measures generic movement decodability. VQD and CONT have the highest means $( 0 . 9 6 0 \pm 0 . 0 3 0$ and $0 . 8 9 9 \pm 0 . 0 9 1 )$ , while AD reaches $0 . 6 0 5 \pm 0 . 1 0 3 .$ . Thus, VQD and CONT make generic motion information more accessible, whereas AD has higher motion-quadrant decodability than the other discrete tokenizers. Because the representations differ in temporal pairing, these probes assess local-motion decodability rather than provide a controlled comparison of future-motion prediction. They measure structural retention, not cross-dataset video-understanding utility. The within-dataset readingstate probe remains a secondary diagnostic in the supplement.

## 4.3. Tokenizer sensitivity

The default AD setting is treated as a fixed reference as opposed to an optimized endpoint. The full multiseed audit is reported in the supplement. Grid size shows a collapse– fragmentation trade-off. A coarse 12×9 grid has low degradation $( 0 . 0 3 \pm 0 . 0 4 ) $ and high order sensitivity, but concentrates over 0.96 of AEA and 0.98 of RITW token mass in the top 10 tokens. A finer 32×24 grid uses more vocabulary but transfers worse than the default and has lower order sensitivity. Removing event type lowers degradation $( 0 . 0 2 \pm 0 . 0 7 )$ but sharply reduces order sensitivity $( 0 . 0 6 \pm 0 . 0 4 )$ , so event conditioning is retained. At the default context length, the 40 ms stride has the highest variance and worst mean degradation, while 120 ms remains competitive but does not dominate the default. This does not contradict the temporal sweep below, where 40 ms becomes useful for $\mathrm { R I T W } { \scriptstyle \to } \mathrm { A E A }$ only with a longer context.

![](images/7e77de4254058aa9552d23ef09626096d8b1a38344cc0ad87a4c852d19cdd575.jpg)

![](images/f760c5bbc434bd4f4561bf1aba3103587a62187805c76223c8f0f952f3b5d78f.jpg)

![](images/70374bd6c4889d0922ed225b31378ada35e36a008b0b31d233c9f46c76a63345.jpg)

D. Order sensitivity  
![](images/68a16fda4f6e9a444eb3fe6a5b68b90fdfea5a27770db29eadcd2541a94337bf.jpg)

E. Bidirectional probe diagnostics  
![](images/566d42906b8f91d1d346f734076898fec3532736d82f3fb495a17b9bbb3b6b2a.jpg)  
Figure 1. Transfer and structure across gaze representations on the bidirectional AEA↔RITW benchmark. Panels A–D summarize the hand-designed tokenizer comparison. Panel E shows the matched context-16 probe rerun with source-specific K = 128 VQD and CONT. $\mathrm { E V T ^ { \dagger } }$ uses AEA→RITW only in Panels A, C, and D. Panel B is NA because the RITW→AEA stream collapses to one deterministic token, making bidirectional degradation undefined. Abbreviations follow Sec. 3.2.

## 4.4. Temporal scale analysis

The best temporal setting depends on transfer direction. For AEA→RITW, a stride of 120 ms with context length 32 gives the lowest OOD perplexity $( 1 . 2 1 \pm 0 . 0 7 )$ . For $\mathrm { R I T W } { \scriptstyle \to } \mathrm { A E A }$ , the best setting uses a 40 ms stride and context length 64 $( 1 . 0 2 2 \pm 0 . 0 0 4 )$ . There is therefore no single best temporal scale.

Equal effective temporal horizon does not imply equal transfer. At a 2560 ms horizon, AEA→RITW obtains OOD perplexity $1 . 2 4 \pm 0 . 1 1$ with (80 ms, 32) but $5 . 2 7 \pm 1 . 0 7$ with (40 ms, 64). In the reverse direction, (40 ms, 64) gives $1 . 0 2 2 { \pm } 0 . 0 0 4$ , compared with $1 . 2 4 \pm 0 . 0 3$ for (160 ms, 16). Token granularity and context length therefore contribute independently.

Follow-up diagnostics suggest two mechanisms. Moderate stride-level smoothing improves the AEA token distribution, increasing entropy from 2.43 at 40 ms to 3.29 at 120 ms while reducing top-10 concentration from 0.94 to 0.72. Smoothed n-gram analysis shows modest gains beyond the bigram level for RITW, whereas AEA saturates earlier. These results are consistent with respective compression and aggregation effects, making the optimal combination direction-dependent. The full heatmap and matched-horizon comparisons are reported in the supplement.

## 4.5. Comparison to continuous and VQ baselines

The continuous autoregressive baseline is evaluated in a different metric family from the discrete token models. Across three seeds, AEA→RITW gives IND MSE $0 . 6 4 \pm 0 . 1 1$ and OOD MSE $0 . 1 6 \pm 0 . 0 4 .$ , with MSE degradation −0.74 ± 0.07. However, the corresponding OOD $R ^ { 2 }$ is close to zero and slightly negative $( - 0 . 0 4 \pm 0 . 1 7 )$ , despite positive IND $R ^ { 2 } \left( 0 . 3 6 \pm 0 . 0 2 \right)$ , consistent with a change in target-domain variance and scale. In the reverse direction, $\mathrm { R I T W } { \scriptstyle \to } \mathrm { A E A }$ is much harder: OOD MSE rises to $5 . 3 7 \pm 1 . 4 6$ and MSE degradation reaches $7 . 5 9 \pm 2 . 5 7$ , while OOD $R ^ { 2 }$ remains positive but lower than IND $R ^ { 2 } \left( 0 . 1 8 \pm 0 . 0 4 \mathrm { v s . 0 . 3 7 \pm 0 . 1 3 } \right)$ These results still make the continuous model a useful predictive baseline, but not a direct replacement for the discrete transfer table, because it is evaluated with continuous regression metrics and does not expose token-level transition structure.

The VQD baseline is viable but codebook-policy sensitive. Our main VQD setting uses a source-specific $K \ = \ 1 2 8$ codebook, which has the lowest mean degradation over $\mathrm { A E A } {  } \mathrm { R I T W }$ among the tested VQ policies $( - 0 . 0 7 \pm 0 . 1 0 )$ . The AEA-reuse policy remains competitive $( - 0 . 0 4 \pm 0 . 1 4 )$ , while the pooled codebook is weaker in transfer $( 0 . 0 3 \pm 0 . 0 2 ) $ and almost eliminates motif overlap despite using nearly all centroids. Its bigram/trigram Jaccard values fall to $0 . 0 0 5 \pm 0 . 0 0 3$ and $0 . 0 0 3 \pm 0 . 0 0 0$ compared with $0 . 2 2 7 \pm 0 . 0 2 2$ and $0 . 0 3 4 \pm 0 . 0 1 1$ for AEA reuse. Thus, VQ codebook policy can reduce distributional mismatch without preserving reusable local token structure. Source-specific fitting has the lowest mean degradation, whereas AEA reuse preserves stronger motif overlap. This reinforces that VQ transfer performance and local-structure preservation are not the same criterion. We treat VQD as a useful learned-motion baseline, but its structural interpretation depends on codebook provenance. The full codebookpolicy results are reported in the supplement.

## 4.6. EGTEA: event construction as a representation boundary

EGTEA tests whether motion-token structure survives changes in device, export format, and event construction. We compare degree-aware I-VT, vendor-native fixation/saccade events, and frame-span events under fixed context 8, minimum sequence length 20, three seeds, identical session splits, uniform session sampling, and matched persession window caps. All interfaces retain the same 86 sessions.

The resulting streams differ substantially. I-VT retains 61,587 events with median sequence length 571 and entropy 5.03 bits. Native events retain 473,683 events with median length 4,090.5 and entropy 3.97 bits. Frame-span construction emits 1,125,577 fixation-typed frame events, but its sequences are much shorter (median 35), more concentrated (top-10 share 0.93), and contain more zero-motion tokens (0.065).

Native events have the lowest target-domain regret for transfer into $\mathrm { E G T E A : 0 . 1 7 2 \pm 0 . 0 2 1 }$ bits/token from AEA and $0 . 2 9 7 { \scriptstyle \pm 0 . 0 3 6 }$ from RITW. The corresponding I-VT regrets are 0.578±0.014 and $1 . 2 6 8 { \pm } 0 . 0 4 5$ , while frame-span regrets are $1 . 0 8 5 { \pm } 0 . 0 1 7$ and $1 . 2 0 9 { \pm } 0 . 0 7 1$ . When EGTEA is the source, native-event regret is $0 . 0 8 5 \pm 0 . 0 1 7$ into AEA and $- 0 . 0 1 5 { \pm } 0 . 0 1 0$ into RITW. I-VT $\mathrm { g i v e s - 0 . 1 4 9 { \pm } 0 . 0 6 1 }$ and $- 0 . 0 2 4 \pm 0 . 0 0 8$ , whereas frame-span regret rises to $5 . 8 0 6 \pm 0 . 1 2 9$ and $6 . 0 4 8 \pm 0 . 1 2 5$ . Negative regret means that the source-trained model scores below the finite targettrained reference under this protocol. It should not be interpreted as outperforming an oracle.

The motif diagnostics show a compatible distinction. I-VT bigram/trigram Jaccard@50 is 0.648/0.381 with AEA and 0.641/0.531 with RITW. Native-event overlap is 0.464/0.276 with AEA and 0.746/0.746 with RITW. Frame-span overlap is zero for both datasets, and all target bigram and trigram mass is unseen across the frame-span boundary.

These results isolate the downstream protocol but not the defining properties of each interface. The interfaces necessarily differ in event boundaries, temporal units, labels, sequence lengths, and event conditioning. The conclusion is therefore not that one interface is universally preferable, but that the upstream definition of an event materially changes the structure and transfer behavior exposed to the same tokenizer.

## 5. Discussion

Representation choice determines what transfers. The main result of this paper is that the choice of discretization determines which structure survives transfer. In the $\mathrm { A E A } {  } \mathrm { R I T W }$ benchmark, angular motion tokens provide a useful hand-designed compromise. They are more informative than event labels, less tied to absolute spatial state, and compact enough for reuse across related egocentric datasets. Against VQD, AD has lower target-domain regret for AEA→RITW, while the reverse direction is inconclusive. EVT shows why low perplexity alone can be misleading: its near-deterministic stream is easy to predict but geometrically empty. In this compatible setting, AD captures local motion with reduced dependence on absolute gaze position under a shared event and coordinate interface, while remaining coarse enough to avoid severe token-space fragmentation.

Expressiveness does not guarantee transfer. The alternative motion baselines illustrate how different representation choices affect transfer. The evaluated SO(3) tokenizer is geometrically expressive, but jointly quantizing rotation angle and axis direction fragments local motion structure—bigram and trigram support nearly vanishes across AEA/RITW, and the angle-only ablation recovers much stronger transfer. VQD shows a different alignment problem: the learned codebook must partition motion space in a way that remains structurally meaningful across domains. A pooled codebook can use centroids efficiently while destroying motif overlap, so codebook usage alone is not enough. In both cases, predictive fit and structural reuse diverge: SD fragments the motion vocabulary despite its geometric expressiveness, while VQD can smooth transfer without preserving reusable local transitions. These comparisons evaluate complete representations: AD and VQD differ in temporal pairing and event conditioning as well as quantization, so they do not isolate the quantizer alone.

Temporal scale is part of the representation. The temporal sweep shows that token scale is not just a downstream model hyperparameter. Equal effective temporal horizon does not imply equal transfer: AEA→RITW benefits from moderate stride-level smoothing, while RITW→AEA prefers finer tokens with longer context. Mechanism analyses suggest a compression effect for AEA, where moderate stride increases token entropy and reduces concentration, and a weaker aggregation effect for RITW, where smoothed n-gram analysis indicates slightly more usable longer-context structure. We do not test scene context causally, so the safest conclusion is that different gaze regimes may require different temporal compression before their structure becomes predictively useful under transfer.

Event interfaces are part of the representation. The matched EGTEA comparison strengthens the eventinterface result. Native events have the lowest targetdomain regret for transfer into EGTEA, while degree-aware I-VT retains substantial motif overlap but shows higher regret into EGTEA. Frame-span events have zero local motif overlap and fail severely when EGTEA is used as the source. Because context, splits, optimizer updates, session sampling, and window caps are controlled, these differences cannot be attributed to the earlier protocol mismatch. They still reflect the complete interfaces, which differ intrinsically in boundaries, temporal units, labels, sequence lengths, and event conditioning. Sharing a motion vocabulary is therefore insufficient. The upstream interface determines which transitions the tokenizer receives.

Practical implications. Motion-based gaze tokens provide a compact input for temporal-attention models, but tokenizer choice remains part of model design. Token streams should be checked for collapse, order sensitivity, and crossdomain motif mismatch, because poor transfer may reflect either a different gaze regime or an incompatible event interface.

Limitations and future work. The study covers three egocentric datasets, so its conclusions concern event-based egocentric gaze transfer under the tested interfaces rather than gaze representation generally. The clearest comparative evidence comes from AEA and RITW, which share the Project Aria device family. Their common sensor geometry, coordinate conventions, and preprocessing assumptions likely make transfer easier than across arbitrary eye trackers or stimulus regimes. The datasets also differ in sequence length and available training windows, which can interact with context length and order-manipulation controls. We address this through matched protocols and temporal scale analyses, but larger balanced egocentric corpora are needed to separate dataset size, task regime, and sensor effects more cleanly. The continuous and discrete baselines use different metric families and should therefore be interpreted as complementary rather than directly interchangeable. Finally, the supervised probes are lightweight diagnostics of structural retention and behavioral decodability, not evidence of cross-dataset video-understanding utility. Stronger application-level tests should combine gaze tokens with egocentric video for activity recognition, taskstate prediction, gaze-conditioned video understanding, or attention-guidance systems.

Privacy and Ethical Considerations. Discrete motion tokens should not be treated as anonymization. They remove some spatial precision and may make exact coordinate- or stimulus-level reconstruction harder, but they intentionally preserve temporal structure. That structure can still encode task, context, user state, ability, or interaction patterns. Tokenization may therefore reduce some risks of raw gaze traces while transforming others into sequencelevel behavioral signatures.

## 6. Conclusion

This paper studies gaze tokenization as a representation problem and asks which abstraction remains useful for egocentric event-based transfer under dataset shift. In the AEA↔RITW benchmark, AD has lower target-domain regret than VQD for AEA→RITW, while RITW→AEA remains inconclusive under paired bootstrap. AD preserves useful order sensitivity and motif overlap under the shared event and coordinate interface, whereas continuous and VQD representations retain complementary motion structure in the probes. Temporal and tokenizer-sensitivity analyses show that token granularity, context length, grid design, event conditioning, and zero/out-of-range handling all shape what structure becomes learnable. A matched EGTEA comparison further shows that degree-aware I-VT, vendor-native, and frame-span interfaces expose markedly different transfer and motif structure to the same tokenizer, with frame-span events failing severely when used as the source. Motion-based tokenization is therefore a compact and interpretable option for compatible egocentric gaze streams, provided that tokenizer design, event interfaces, and retained behavioral structure are explicitly considered. The broader lesson is that cross-dataset gaze modeling requires aligning not only models, but also the behavioral units and token vocabularies through which gaze is exposed to those models.

## References

[1] Nicola C Anderson, Walter F Bischof, Kaitlin EW Laidlaw, Evan F Risko, and Alan Kingstone. Recurrence quantifica-

tion analysis of eye movements. Behavior research methods, 45(3):842–856, 2013. 2

[2] Nicola C Anderson, Fraser Anderson, Alan Kingstone, and Walter F Bischof. A comparison of scanpath comparison methods. Behavior research methods, 47(4):1377–1392, 2015. 2

[3] A Terry Bahill, Michael R Clark, and Lawrence Stark. The main sequence, a tool for studying human eye movements. Mathematical biosciences, 24(3-4):191–204, 1975. 2

[4] Robert Bixler and Sidney D’Mello. Automatic gaze-based user-independent detection of mind wandering during computerized reading. User Modeling and User-Adapted Interaction, 26(1):33–68, 2016. 1

[5] Stanley F Chen and Joshua Goodman. An empirical study of smoothing techniques for language modeling. Computer Speech & Language, 13(4):359–394, 1999. 4

[6] Tim Chuk, Antoni B Chan, and Janet H Hsiao. Understanding eye movements in face recognition using hidden markov models. Journal ofvision, 14(11):8–8, 2014. 2

[7] Antoine Coutrot, Janet H Hsiao, and Antoni B Chan. Scanpath modeling and classification with hidden markov models. Behavior research methods, 50(1):362–379, 2018. 2

[8] Filipe Cristino, Sebastiaan Mathot, Jan Theeuwes, andˆ Iain D Gilchrist. Scanmatch: A novel method for comparing fixation sequences. Behavior research methods, 42(3): 692–700, 2010. 2

[9] Richard Dewhurst, Marcus Nystrom, Halszka Jarodzka, Tom¨ Foulsham, Roger Johansson, and Kenneth Holmqvist. It depends on how you look at it: Scanpath comparison in multiple dimensions with multimatch, a vector-based approach. Behavior research methods, 44(4):1079–1100, 2012. 2

[10] Jakob Engel, Kiran Somasundaram, Michael Goesele, Albert Sun, Alexander Gamino, Andrew Turner, Arjang Talattof, Arnie Yuan, Bilal Souti, Brighid Meredith, et al. Project aria: A new tool for egocentric multi-modal ai research. arXiv preprint arXiv:2308.13561, 2023. 2

[11] Alireza Fathi, Yin Li, and James M Rehg. Learning to recognize daily actions using gaze. In European Conference on Computer Vision, pages 314–327. Springer, 2012. 1

[12] Anshul Gupta, Samy Tafasca, Arya Farkhondeh, Pierre Vuillecard, and Jean-Marc Odobez. Mtgs: A novel framework for multi-person temporal gaze following and social gaze prediction. Advances in Neural Information Processing Systems, 37:15646–15673, 2024. 2

[13] Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Matusik, and Antonio Torralba. Gaze360: Physically unconstrained gaze estimation in the wild. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6912–6921, 2019. 2

[14] Reinhard Kneser and Hermann Ney. Improved backing-off for m-gram language modeling. In 1995 international conference on acoustics, speech, and signal processing, pages 181–184. IEEE, 1995. 4

[15] Thomas C Kubler, Colleen Rothe, Ulrich Schiefer, Wolfgang¨ Rosenstiel, and Enkelejda Kasneci. Subsmatch 2.0: Scanpath comparison and classification based on subsequence frequencies. Behavior research methods, 49(3):1048–1064, 2017. 2

[16] Quentin Laborde, Axel Roques, Allan Armougum, Nicolas Vayatis, Ioannis Bargiotas, and Laurent Oudre. Vision toolkit part 3. scanpaths and derived representations for gaze behavior characterization: a review. Frontiers in Physiology, 16: 1721768, 2026. 2

[17] Gen Li, Yutong Chen, Yiqian Wu, Kaifeng Zhao, Marc Pollefeys, and Siyu Tang. Egom2p: Egocentric multimodal multitask pretraining. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10830– 10843, 2025. 1

[18] Yin Li, Miao Liu, and James M. Rehg. In the eye of the beholder: Gaze and actions in first person video. IEEE Trans. Pattern Anal. Mach. Intell., 45(6):6731–6747, 2023. 1, 2, 3

[19] Yiwei Li, Zihao Wu, Yanjun Lv, Hanqi Jiang, Weihang You, Zhengliang Liu, Dajiang Zhu, Xiang Li, Quanzheng Li, Tianming Liu, et al. Thinking with gaze: Sequential eyetracking as visual reasoning supervision for medical vlms. arXiv preprint arXiv:2603.06697, 2026. 2

[20] Zhaoyang Lv, Nicholas Charron, Pierre Moulon, Alexander Gamino, Cheng Peng, Chris Sweeney, Edward Miller, Huixuan Tang, Jeff Meissner, Jing Dong, et al. Aria everyday activities dataset. arXiv preprint arXiv:2402.13349, 2024. 1, 2

[21] Marcus Nystrom and Kenneth Holmqvist. An adaptive al-¨ gorithm for fixation, saccade, and glissade detection in eyetracking data. Behavior research methods, 42(1):188–204, 2010. 3

[22] Tim Rolff, Jurik Karimian, Niklas Hypki, Susanne Schmidt, Markus Lappe, and Frank Steinicke. Tokenization of gaze data. arXiv preprint arXiv:2503.22145, 2025. 2

[23] Samy Tafasca, Anshul Gupta, and Jean-Marc Odobez. Sharingan: A transformer architecture for multi-person gaze following. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2008–2017, 2024. 2

[24] Charig Yang, Samiul Alam, Shakhrul Iman Siam, Michael J Proulx, Lambert Mathias, Kiran Somasundaram, Luis Pesqueira, James Fort, Sheroze Sheriffdeen, Omkar Parkhi, et al. Reading recognition in the wild. arXiv preprint arXiv:2505.24848, 2025. 1, 2

[25] Alfred L Yarbus. Eye movements and vision. Eye movements and vision., page 171, 1967. 1

[26] Xucong Zhang, Yusuke Sugano, Mario Fritz, and Andreas Bulling. Appearance-based gaze estimation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4511–4520, 2015. 2

[27] Xucong Zhang, Yusuke Sugano, Mario Fritz, and Andreas Bulling. Mpiigaze: Real-world dataset and deep appearancebased gaze estimation. IEEE transactions on pattern analy sis and machine intelligence, 41(1):162–175, 2017. 2

[28] Yang Zheng, Yanchao Yang, Kaichun Mo, Jiaman Li, Tao Yu, Yebin Liu, C Karen Liu, and Leonidas J Guibas. Gimo: Gaze-informed human motion prediction in context. In Eu ropean Conference on Computer Vision, pages 676–694. Springer, 2022. 1

[29] Yuchen Zhou, Linkai Liu, and Chao Gou. Learning from observer gaze: Zero-shot attention prediction oriented by

human-object interaction recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28390–28400, 2024. 1