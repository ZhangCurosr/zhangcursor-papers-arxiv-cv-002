# OMNIGLIMPSE: ASYMMETRIC IDENTITY MEMORY FOR ONLINE MULTI-VIEW TRACKING

Rangya Zhang<sup>1∗</sup>, Jiaping Xiao<sup>1∗</sup>, Yujing Shang<sub>Mingsheng Li1, Mir Feroskhan</sub> 1∗ Yuhang Zhang<sup>1</sup>, ,<sub>1†</sub>

<sup>1</sup>Nanyang Technological University

## ABSTRACT

Multi-view tracking extends visual coverage beyond a single camera, providing complementary observations when targets are occluded or leave the view. This raises a question: how can observations from different views be matched to the correct identities without offline scene-specific training or camera calibration? Progress toward this goal remains limited because many offline formulations assume such deployment-specific knowledge, while online adaptation can amplify association errors. We study this problem as online multi-view identity maintenance (OMIM): targets are initialized once per view, with no further identity labels provided after each initialization. To this end, we introduce OmniGlimpse, an asymmetric propose–verify framework that separates adaptive association from stable identity consolidation. At the representation level, short-term local and cross-view evidence is mapped into target-specific adaptive spaces to propose associations, whereas long-term identity distributions are maintained for verification in a canonical space defined by a frozen encoder. At the update level, short-term evidence may guide current associations, whereas, after bootstrapping, only verified observations may update persistent memory or drive adaptation. Under a unified target-specified evaluation protocol, OmniGlimpse improves over the best evaluated online baseline for each metric by 9.1 IDF1 and 22.1 CVIDF1 points on M3Track. The corresponding gains on MvMHAT are 20.3 and 26.2 points. Despite using no offline scene-specific training, it outperforms offline-trained references in IDF1 and remains competitive in cross-view association. These results support an approach to online identity maintenance that adapts to changing appearances while consolidating reliable evidence into stable identity anchors.

![](images/937c592c13432f996733f8b4a1b157d738bdf1bf12ed867c9ae0bda1a08ea95b.jpg)  
Figure 1: Overview of OmniGlimpse. Top: targets are initialized once per view and maintained through adaptive association, canonical verification, and verified updates. Bottom: OmniGlimpse supports cross-view recovery, appearance adaptation, and resistance to distractors.

## 1 INTRODUCTION

Multi-view tracking maintains identity continuity beyond a single camera and supports long-horizon perception in surveillance Xie et al. (2024), sports analysis Cui et al. (2023), and traffic understanding Liu et al. (2024). Multiple views provide complementary observations of the targets but also introduce new ambiguity in identity matching. The tracker must distinguish each target from visually similar distractors despite large viewpoint and appearance changes. Multi-view trackers associate observations through appearance, trajectories, or geometry Gan et al. (2021); Cheng et al. (2023a), with online variants updating tracks and appearance memories as observations arrive Xie et al. (2024); Specker (2024). A related line of target-specific online methods adapts the representation to changes in a designated target’s appearance Ye et al. (2022; 2024).

Deploying these capabilities together in unseen multi-view scenes exposes three connected challenges. (i) Deployment requirements. Multi-view multi-target (MVMT) trackers often learn appearance models offline Gan et al. (2021); Fan et al. (2025), while geometry-based approaches additionally require calibration or world coordinates Specker (2024), restricting deployment when these geometric priors are unavailable. (ii) Uncertain cross-view evidence. Cross-view observations can compensate for unreliable local evidence, but view-dependent appearance gaps and asymmetric observation noise Amosa et al. (2023) make these cues themselves uncertain. (iii) Self-reinforcing errors. When recent identity assignments are reused for online adaptation, a wrong pseudo-label can enter memory, alter the representation, and bias subsequent matches. Together, these challenges make the problem more than online cross-view association: the tracker must also determine which observations are reliable enough to change its persistent identity state.

We refer to this as online multi-view identity maintenance (OMIM). Each target is initialized with a bounding box at its first valid appearance in each view. Its identity is then maintained across views without further identity labels. To address this, we introduce OMNIGLIMPSE, an asymmetric propose–verify framework that separates adaptive association from stable identity consolidation. Its design is asymmetric in both update permission and representation space. First, short-term evidence can only propose or guide associations. After initialization and bootstrapping, only observations consistent with the long-term identity distributions may update persistent memory and target-specific adaptive spaces. Second, short-term local and cross-view evidence is matched in the adapters’ adaptive spaces to propose associations, while per-target identity distributions serve as anchors for verification in a canonical space defined by a frozen encoder. This asymmetric design enables OmniGlimpse to adapt to changing appearances while preserving stable verification and supervision. Experiments on M3Track Fan et al. (2025) and MvMHAT Gan et al. (2021) show improved identity maintenance over online baselines, with ablations supporting the design.

Our work makes three main contributions:

• We introduce OmniGlimpse for online multi-view identity maintenance, where targets are initialized once per view and maintained across views without further identity labels. The framework combines cross-view evidence with target-specific online adaptation to track changing appearances without offline scene-specific training or camera calibration.

• We develop an asymmetric propose–verify design with update permissions and representation spaces. Short-term evidence proposes associations. After bootstrapping, long-term verification controls which observations may update persistent memory and adaptive representations. A frozen canonical space anchors the evolving target-specific adapters.

• We evaluate OmniGlimpse on M3Track and MvMHAT under a unified target-specified protocol. It outperforms evaluated online baselines in IDF1 and CVIDF1 and remains competitive with offline-trained references. Ablations support the benefits of online adapter learning and verification-gated updates within the complete system.

## 2 RELATED WORK

Single-view online identity tracking. Online MOT maintains identities by associating detections with motion and appearance cues Maggiolino et al. (2023); Shim et al. (2025). ByteTrack Zhang et al. (2022b) exploits low-confidence detections to recover fragmented trajectories, while OC SORT Cao et al. (2023) improves motion-based association under occlusion and non-linear motion.

Template updating Zhou et al. (2026) and test-time adaptation Zaveri et al. (2025); Segu et al. (2023) address changes in appearance or deployment conditions. Target-specific methods such as Following Closely Ye et al. (2022) and OCLReID Ye et al. (2024) update the target appearance model. OCLReID further uses short-term and long-term experiences to optimize the feature extractor online. Such adaptation improves responsiveness, but recent identity assignments also become supervision for subsequent updates. An incorrect assignment can therefore alter the appearance model used for future matching, allowing a temporary association error to persist through later updates.

Multi-view identity association. Multi-view tracking extends identity maintenance by using observations from cameras when local evidence becomes unreliable. One approach establishes crossview correspondence from appearance and trajectory cues Huang et al. (2023); Yamane et al. (2025). MvMHAT Gan et al. (2021) considers tracking and cross-view association, while ReST Cheng et al. (2023a) models spatial-temporal relations through graph-based association. Benchmarks MT-MMC Woo et al. (2024) and M3Track Fan et al. (2025) extend evaluation to realistic scenes. When appearance is ambiguous, MUTR3D Zhang et al. (2022a), ADA-Track Ding et al. (2024), and OCM-CTrack Specker (2024) exploit 3D queries, geometric cues, or world coordinates for cross-view matching. Although OCMCTrack performs association online, such methods may still depend on offline appearance learning Gan et al. (2021); Fan et al. (2025) or calibration and world-coordinate information Specker (2024), limiting deployment when these priors are unavailable.

Memory and Adaptation in Identity Tracking. Memory has been used in tracking and ReID systems to support identity decisions Videnovic et al. (2026). Templates, tracklet states, or mem-´ ories are maintained to recover targets after occlusion and stabilize appearance matching Aydemir et al. (2025). Person ReID stores identity features or prototypes, providing more stable references than the observation alone Dai et al. (2022); Cheng et al. (2023b). Online adaptation uses unlabeled observations to update caches or refine representations Karmanov et al. (2024); Segu et al. (2023). However, evidence useful for matching may not be reliable enough to supervise updates, while representation changes can make stored features inconsistent with observations. OmniGlimpse addresses these issues through an asymmetric propose–verify design that combines adaptive association with canonical-space verification to govern persistent memory updates and learning after bootstrapping.

## 3 METHOD

## 3.1 PROBLEM SETUP AND OVERVIEW

Our objective is to maintain the identities of a small set of targets of interest $\mathcal { T } = \{ 1 , \dots , M \}$ across multiple views $\mathcal { V } = \{ 1 , \ldots , V \}$ . Each target $m \in \mathcal { T }$ is initialized with a bounding box at its first valid appearance in each view. The target reference images $\mathcal { P } _ { m }$ are crops extracted from these initialization boxes and become available only when the corresponding initialization occurs. No further frame-level identity annotations are provided after initialization in each view.

At time t, view v produces an observation set ${ \mathcal O } _ { t } ^ { ( v ) }$ containing $n _ { t } ^ { ( v ) }$ observations. Using only observations and initialization information available up to $t ,$ the system must infer, for every target m and view $v ,$ the observation currently corresponding to that target:

$$
\mathcal { O } _ { t } ^ { ( v ) } = \big \{ o _ { t , i } ^ { ( v ) } \big \} _ { i = 1 } ^ { n _ { t } ^ { ( v ) } } , \qquad a _ { t } ^ { ( m , v ) } \in \mathcal { O } _ { t } ^ { ( v ) } \cup \{ \varnothing \} .\tag{1}
$$

Here, ∅ denotes that the target is absent, occluded, or cannot be identified with sufficient confidence.   
The objective is to maintain the identity consistency of each target across views and over time.

Given observations and target priors, OmniGlimpse maintains target identities through a closed-loop process of target-specific association, long-term verification, and online adaptation (Fig. 2).

## 3.2 ASSOCIATION AND VERIFICATION

Canonical and adaptive features. Online association must adapt to appearance changes, while cross-view evidence and long-term identity statistics need a consistent feature space across updates. At each time step, an initial detector–tracker produces person observations, each consisting of a bounding box, a person crop, and a local track identity. Following the part-based representation design in OCLReID Ye et al. (2024), a ReID encoder B extracts a global embedding $g _ { 0 } ( o )$ and K part embeddings $\{ g _ { k } ( o ) \} _ { k = 1 } ^ { K }$ from each observation o. All views use the same frozen encoder, while each target–view pair maintains a local adapter $A _ { t } ^ { ( m , v ) }$

![](images/89345ae2220f691bbbd3fca3b1c8cdf7ea261f60e2057fb17eb04c42ad7eef5f.jpg)  
Figure 2: Overview of OmniGlimpse. Short-term evidence supports association in adaptive spaces, while long-term distributions provide canonical-space verification. After initialization and bootstrapping, only admitted observations may update persistent memory and local adapters.

$$
u ( o ) = B ( o ) = \bigl ( g _ { 0 } ( o ) , \ldots , g _ { K } ( o ) \bigr ) , \qquad z _ { t } ^ { ( m , v ) } ( o ) = A _ { t } ^ { ( m , v ) } \bigl ( u ( o ) \bigr ) .\tag{2}
$$

Here, $u ( o )$ denotes the canonical features and $z _ { t } ^ { ( m , v ) } ( o )$ the adaptive features, with $z _ { t , k } ^ { ( m , v ) } ( o )$ denoting the global or part cue k. The part features are accompanied by visibility scores $w _ { k } ( o ) \in [ 0 , 1 ]$ and an orientation head predicts $\omega ( o ) \in$ {front, back} for orientation-aware scoring.

Memory banks store canonical features, and views exchange these features rather than local adapter outputs. For short-term association, both local and received cross-view features are mapped through the current local adapter. Prior references and long-term identity distributions instead use the canonical features for matching and verification. Online learning updates the local adapters without changing B, preserving the coordinate system used for communication and long-term verification. The identity distributions can still accumulate verified evidence within this fixed space.

Candidate association. For each target–view pair $( m , v )$ , candidate observations are evaluated using initialization references and adaptive ridge classifiers. The prior score measures global and part-level similarity in the canonical space, while the classifiers evaluate adaptive features under current appearance conditions. Prior evidence supports initialization and recovery; once the classifiers are available, their confidence provides the local association score $s _ { \mathrm { l o c a l } } ^ { ( m , v ) } ( o )$

The active support bank contains local target samples, temporally eligible cross-view samples, and initialization references. All support features are mapped through the current local adapter before meets a discriminativeness criterion and global evidence otherwise. Cross-view evidence therefore contributes through the support bank rather than a separate additive score.

Memory assistance is activated when the current binding is missing, uncertain, or contested. For these candidates, the association score combines local confidence and memory support:

$$
s _ { \mathrm { a s s o c } } ^ { ( m , v ) } ( o ) = ( 1 - \lambda _ { t } ^ { ( m , v ) } ) s _ { \mathrm { l o c a l } } ^ { ( m , v ) } ( o ) + \lambda _ { t } ^ { ( m , v ) } s _ { \mathrm { m e m } } ^ { ( m , v ) } ( o ) .\tag{3}
$$

Both scores are bounded in [0, 1], and $\lambda _ { t } ^ { ( m , v ) } = \lambda _ { \mathrm { m e m } } \in [ 0 , 1 ]$ when assistance is active and support is available; otherwise $\lambda _ { t } ^ { ( m , v ) } = 0$ . These bounded scores are not calibrated identity probabilities.

For targets in the view, we first retain uncontested bindings that satisfy the tracking conditions. For the remaining targets, eligible target–observation pairs are ranked by $s _ { \mathrm { a s s o c } }$ and assigned greedily, allowing each target and observation to participate in at most one pair. Candidates below the acceptance threshold are discarded, and overlapping assignments are suppressed. The associations remain proposals for long-term verification; selecting a candidate does not authorize memory consolidation or learning. Scoring, support eligibility, and conflict handling are detailed in Appendix A.1.

Long-term verification. The long-term memory verifies candidate associations by maintaining a consolidated identity distribution for each target. Since target appearance can be multi-modal under viewpoint changes, occlusion, and partial visibility, we represent this distribution with a small set of Gaussian prototypes. Each view v maintains a local target distribution

$$
\mathcal { G } _ { t } ^ { ( m , v ) } = \big \{ ( \mu _ { j } ^ { ( m , v ) } , \sigma _ { j } ^ { 2 , ( m , v ) } , n _ { j } ^ { ( m , v ) } ) \big \} _ { j = 1 } ^ { P _ { t } ^ { ( m , v ) } } , \qquad P _ { t } ^ { ( m , v ) } \leq P ,\tag{4}
$$

where $\mu _ { j }$ is the prototype mean, $\sigma _ { j } ^ { 2 }$ its diagonal variance, and $n _ { j }$ its admitted-sample count. These distributions use normalized canonical global features and can incorporate evidence from other views. They share a feature coordinate system, not necessarily identical distribution parameters.

For a candidate observation $^ { O , }$ let $\bar { u } ( o ) = g _ { 0 } ( o ) / \lVert g _ { 0 } ( o ) \rVert _ { 2 }$ . The verifier measures its agreement with the consolidated distribution using a dimension-normalized squared Mahalanobis distance:

$$
d _ { j } ^ { ( m , v ) } ( \bar { u } ) = \frac { 1 } { D } \sum _ { r = 1 } ^ { D } \frac { ( \bar { u } _ { r } - \mu _ { j , r } ^ { ( m , v ) } ) ^ { 2 } } { \sigma _ { j , r } ^ { 2 , ( m , v ) } + \epsilon } , \qquad \ell _ { \mathrm { l o n g } } ^ { ( m , v ) } ( o ) = \exp \biggl [ - \frac { 1 } { 2 } \operatorname* { m i n } _ { j } d _ { j } ^ { ( m , v ) } ( \bar { u } ( o ) ) \biggr ] .\tag{5}
$$

Here, D is the global feature dimension and $\epsilon > 0$ stabilizes the denominator. The diagonal variance accounts for uncertainty along different feature dimensions, while the minimum selects the bestmatching appearance mode. The resulting consistency score is not a normalized identity probability.

Verification uses the distribution available before admitting the current observation. Once the count $\begin{array} { r } { n _ { t } ^ { ( m , v ) } = \sum _ { i } n _ { j } ^ { ( m , v ) } } \end{array}$ reaches $N _ { \mathrm { m i n } } .$ , a candidate passes if $\ell _ { \mathrm { l o n g } } ^ { ( m , v ) } ( o ) \geq \tau _ { \ell }$ and fails otherwise. Before this point, the result is unavailable, neither failed nor verified. Initialization references seed the distribution, which matures through conservative bootstrap admission (Appendix A.2).

A failed check prevents a new or recovered association from being committed. An established binding may remain an output if it still satisfies the tracking conditions, but the failed observation cannot update persistent memory or train the adapters or classifiers. Passing verification indicates consistency with the identity model; further admission rules govern updates. This separates evidence for current association from evidence allowed to modify persistent identity state.

## 3.3 VERIFICATION-GATED UPDATES

Association outputs and learning samples have different admission requirements. After assignment, an output observation may enter the short-term support bank to assist subsequent proposals, but its presence there does not authorize training or long-term consolidation. Each stored observation retains its admission status. Persistent identity memory and positive training buffers accept only evidence that satisfies the update rule below, apart from initialization and restricted bootstrap.

For a local observation $^ { O , }$ let $V _ { t } ^ { ( m , v ) } ( o ) = 1$ only when the mature long-term verifier returns pass, and zero when it returns fail or unavailable. A positive observation is admitted if

$$
\begin{array} { r l } & { \mathcal { A } _ { + , t } ^ { ( m , v ) } ( o ) = \mathbb { I } \big [ o = a _ { t } ^ { ( m , v ) } \wedge V _ { t } ^ { ( m , v ) } ( o ) = 1 } \\ & { \qquad \wedge \ s _ { \mathrm { a s s o c } } ^ { ( m , v ) } ( o ) \geq \tau _ { a } } \\ & { \qquad \wedge \ \operatorname* { m i n } \{ \Delta _ { \mathrm { t g t } } ( o ) , \Delta _ { \mathrm { o b s } } ( o ) \} \geq \tau _ { \Delta } } \\ & { \qquad \wedge \ T _ { \mathrm { s t a b l e } } ^ { ( m , v ) } ( o ) \geq \tau _ { T } \big ] . } \end{array}\tag{6}
$$

Here, $a _ { t } ^ { ( m , v ) }$ is the final retained assignment, the two margins are computed before allocation, and $T _ { \mathrm { s t a b l e } }$ counts consecutive observed frames with the same target–track binding. Missing observations or a changed binding reset this count. The rule controls long-memory admission and positive samples for both Ridge fitting and adapter learning; it is not a universal condition for emitting an association. Initialization labels provide initial supervision. Before maturity, conservative bootstrap can populate the long-term distribution but cannot authorize ordinary pseudo-label training. Remote evidence requires independent receiver-side admission (Appendix A.2).

All admission decisions are made before any memory or parameter update. We first fix the admitted sample set using the current assignments and pre-update distributions, then use that set for longmemory maintenance and online learning. The Gaussian parameters used as adaptation anchors are cached before these updates, so an observation cannot modify its own anchor within the same step. The updated distributions are used for verification only at the next step.

## 3.4 DISTRIBUTION-ANCHORED ADAPTATION

The propose–verify memory maintains candidate-level identity consistency, but appearance changes can still weaken feature discriminability. We therefore refine the local adapters using admitted memory samples. Each adapter applies a residual transformation to the canonical cues,

$$
z _ { t , k } ^ { ( m , v ) } ( o ) = g _ { k } ( o ) + h _ { \phi _ { t , k } ^ { ( m , v ) } } \big ( g _ { k } ( o ) \big ) , \qquad \widehat { z } _ { t , k } ^ { ( m , v ) } ( o ) = \frac { z _ { t , k } ^ { ( m , v ) } ( o ) } { \| z _ { t , k } ^ { ( m , v ) } ( o ) \| _ { 2 } } .\tag{7}
$$

The residual branches initially output zero, so the normalized adaptive features start from their canonical counterparts. The frozen encoder defines the coordinates, while residual branches learn target-specific corrections. Architecture and sampling details are given in Appendix A.3.

For each target–view pair, the memory system constructs same-view and cross-view triplets $( a , p , n )$ from admitted evidence. The anchor a and positive p represent the same target; negatives come from observations admitted for other specified targets. Unassigned detections and unresolved overlapping observations are excluded rather than assumed to be reliable negatives. All three samples are mapped through the same current local adapter, including those received from other views. Sameview triplets preserve local discrimination, while cross-view positives bring different appearances together within the receiver’s adaptive space. For jointly visible cues, we use the loss

$$
\ell _ { \mathrm { t r i } , k } ( a , p , n ) = \left[ \cos ( \widehat { z } _ { k } ( a ) , \widehat { z } _ { k } ( n ) ) - \cos ( \widehat { z } _ { k } ( a ) , \widehat { z } _ { k } ( p ) ) + \delta _ { \mathrm { t r i } } \right] _ { + } .\tag{8}
$$

Here, target, view, and time indices are omitted for readability. Averaging over available cues and triplets gives $\mathcal { L } _ { \mathrm { s v } }$ and ${ \mathcal { L } } _ { \mathrm { c v } }$ for same-view and cross-view evidence.

Triplet learning separates target observations from distractors. To also regularize adaptation with accumulated identity evidence, we penalize deviations of admitted positive global features from the canonical Gaussian prototypes:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a n c h o r } } = \displaystyle \frac { 1 } { | \mathcal { P } _ { + } | } \sum _ { o \in \mathcal { P } _ { + } } \operatorname* { m i n } _ { j } d _ { j , t } ^ { ( m , v ) } \big ( \widehat { z } _ { t , 0 } ^ { ( m , v ) } ( o ) \big ) , } \\ & { \mathcal { L } _ { \mathrm { a d a p t } } = \mathcal { L } _ { \mathrm { s v } } + \gamma _ { \mathrm { c v } } \mathcal { L } _ { \mathrm { c v } } + \beta _ { \mathrm { d i s t } } \mathcal { L } _ { \mathrm { a n c h o r } } . } \end{array}\tag{9}
$$

Here, $\mathcal { P } _ { + }$ contains the positive observations in the training batch, and $d _ { j , t } ^ { ( m , v ) }$ uses the cached preupdate Gaussian parameters in Eq. 5. The residual initialization makes this penalty compare aligned coordinates at the start of adaptation; the penalty subsequently discourages departures from the canonical identity distribution. It constrains the adaptive global features, whereas inference-time verification continues to use the unmodified canonical features.

During adaptation, Gaussian parameters are detached and gradients update only the local residual branches. After an update, cached canonical features are remapped through the new adapter, and Ridge classifiers are refit using only samples with training permission. The refreshed features and classifiers support the next association step without another backbone pass on stored crops. Buffertriggered refitting and missing-class handling are detailed in Appendix A.1.

Table 1: Selected-target results on M3Track. Offline references use target-selection information, including ground-truth-assisted trajectory selection for CrossMOT and AGW and full-sequence appearance matching for MvMHAT (Appendix B.2). Best and second-best in bold and underline.
<table><tr><td>Method</td><td>HOTA↑</td><td>DetA↑</td><td>AssA↑</td><td>IDF1↑</td><td>MOTA↑</td><td>MOTP↑</td><td>IDs↓</td><td>MDA↑</td><td>CVIDP↑</td><td>CVIDR↑</td><td>CVIDF1↑</td><td>CVMA↑</td></tr><tr><td colspan="9">Offline-trained MVMT references</td><td></td><td></td><td></td></tr><tr><td>MvMHAT</td><td>.5832</td><td>.4407</td><td>.7723</td><td>.6267</td><td>.4641</td><td>.9335</td><td>11</td><td>.1620</td><td>.2914</td><td>.1920</td><td>.2277</td><td>.3204</td></tr><tr><td>CrossMOT</td><td>.5970</td><td>.5540</td><td>.6445</td><td>.7945</td><td>.6628</td><td>.8182</td><td>116</td><td>.5635</td><td>.6150</td><td>.5499</td><td>.5786</td><td>.6335</td></tr><tr><td>AGW</td><td>.4748</td><td>.3928</td><td>.5773</td><td>.6347</td><td>.4630</td><td>.7936</td><td>9</td><td>.2350</td><td>.3374</td><td>.2556</td><td>.2864</td><td>.3717</td></tr><tr><td colspan="10">Online baselines</td><td></td><td></td><td></td></tr><tr><td>OCLReID</td><td>.5862</td><td>.4810</td><td>.7173</td><td>.7424</td><td>.5795</td><td>.7972</td><td>18</td><td>.3290</td><td>.4480</td><td>.3552</td><td>.3918</td><td>.4586</td></tr><tr><td>OC-SORT+RPF ByteTrack</td><td>.5071</td><td>.3904</td><td>.6614</td><td>.6508</td><td>.4603</td><td>.7960</td><td>2</td><td>.2678</td><td>.3771</td><td>.2892</td><td>.3237</td><td>.3984</td></tr><tr><td>BoT-SORT+ReID</td><td>.4664 .4651</td><td>.3610 .3559</td><td>.6073 .6128</td><td>.5872 .5944</td><td>.3967</td><td>.8016</td><td>8 5</td><td>.2582 .2892</td><td>.3282 .4036</td><td>.2687 .2955</td><td>.2916 .3375</td><td>.3607 .3889</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>.4013</td><td>.7980</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours (Full)</td><td>.6438</td><td>.5861</td><td>.7102</td><td>.8333</td><td>.7217</td><td>.7945</td><td>28</td><td>.5503</td><td>.6576</td><td>.5792</td><td>.6129</td><td>.6677</td></tr></table>

Table 2: Selected-target results on MvMHAT. Offline references use information for target selection: CrossMOT uses ground-truth-assisted trajectory selection; MvMHAT and AGW use full-sequence appearance matching (Appendix B.2). Best and second-best results are shown in bold and underline.
<table><tr><td>Method</td><td>HOTA↑</td><td>DetA↑</td><td>AssA↑</td><td>IDF1↑</td><td>MOTA↑</td><td>MOTP↑</td><td>IDs↓</td><td>MDA↑</td><td>CVIDP↑</td><td>CVIDR↑</td><td>CVIDF1↑</td><td>CVMA↑</td></tr><tr><td colspan="9">Offline-trained MVMT references</td><td></td><td></td><td></td></tr><tr><td>MvMHAT</td><td>.3646</td><td>.1786</td><td>.7443</td><td>.3012</td><td>.1448</td><td>.9556</td><td>1</td><td>.0500</td><td>.0522</td><td>.0505</td><td>.0513</td><td>.2354</td></tr><tr><td>CrossMOT AGW</td><td>.5217</td><td>.4448 .4352</td><td>.6130 .7628</td><td>.6901</td><td>.4874</td><td>.8196</td><td>17</td><td>.5626</td><td>.6380</td><td>.5819</td><td>.6078</td><td>.7190 .5114</td></tr><tr><td></td><td>.5761</td><td></td><td></td><td>.5875</td><td>.3518</td><td>.9993</td><td>6</td><td>.3448</td><td>.4687</td><td>.3759</td><td>.4131</td><td></td></tr><tr><td colspan="10">Online baselines</td><td></td><td></td><td></td></tr><tr><td>OCLReID</td><td>.4644 .3444</td><td>.3060</td><td>.7133</td><td>.5292</td><td>.3290</td><td>.8403</td><td></td><td>.2280</td><td>.2975</td><td>.2454</td><td>.2676</td><td>.4245</td></tr><tr><td>OC-SORT+RPF ByteTrack</td><td>.4516</td><td>.2008</td><td>.6024</td><td>.3984</td><td>.2319</td><td>.7833</td><td>3-1</td><td>.0786</td><td>.1112</td><td>.0868</td><td>.0964</td><td>.2878 .4494</td></tr><tr><td>BoT-SORT+ReID</td><td>.4198</td><td>.3112 .2742</td><td>.6651 .6509</td><td>.5130 .4661</td><td>.3104</td><td>.8448 .8514</td><td>9 4</td><td>.2719 .2114</td><td>.3330 .2764</td><td>.2875 .2278</td><td>.3068 .2481</td><td>.4007</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>.2754</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Ours (Full)</td><td>.5820</td><td>.4863</td><td>.7135</td><td>.7323</td><td>.5060</td><td>.8168</td><td>36</td><td>.5189</td><td>.6053</td><td>.5408</td><td>.5685</td><td>.6601</td></tr></table>

## 4 EXPERIMENTS

Our experiments address two questions: (1) Can OmniGlimpse maintain target identities across views without identity labels after initialization? (2) Can identity verification support online adaptation while limiting the propagation of association errors? We examine its behavior in sequences.

## 4.1 EXPERIMENTAL SETUP

Benchmarks and protocol. M3Track Fan et al. (2025) is our main benchmark, providing synchronized RGB and infrared streams from two moving-camera views. We use only the visible light streams and exclude extremely low-light nighttime sequences, yielding 36 single-view RGB sequences for training offline references and 38 for testing, corresponding to 19 paired-view test instances. MvMHAT Gan et al. (2021) provides 3–4 synchronized views per video group; following its original split, we use 13 groups for training offline references and 13 for testing. In the online protocol, each target is initialized with a bounding box at its first annotated appearance in each view, with no subsequent frame-level identity labels. All methods are evaluated on the same selected identities; offline target-selection procedures are detailed in Appendix B.2.

Baselines. We compare with online methods and offline-trained references. Online baselines are OCLReID Ye et al. (2024), OC-SORT+RPF Cao et al. (2023), ByteTrack Zhang et al. (2022b), and BoT-SORT+ReID Aharon et al. (2022). They receive initialization boxes but no scene-specific training and run independently per view without cross-view information exchange. Corresponding targets share an exported label only to normalize the identity namespace, providing no crossview association or recovery cues. OCLReID and OC-SORT+RPF retain their target-identification mechanisms, whereas ByteTrack and BoT-SORT follow the tracker identity bound at initialization. Offline references are MvMHAT Gan et al. (2021), CrossMOT Hao et al. (2024), and AGW Ye et al. (2021). Since their target selection may use full-sequence information, including GT-assisted trajectory selection, we report them as references rather than identical-condition baselines.

Table 3: Ablation study on 19 non-dark M3Track test instances. The upper block shows incremental variants and the lower block full-model controls (Appendix B.3). V-Upd. denotes verification-gated updates; Adaptive Query uses adapted features. Best and second-best are in bold and underline.
<table><tr><td>Variant</td><td>Rel.</td><td>Cross</td><td>Adapt.</td><td>Gauss.</td><td>V-Upd.</td><td>HOTA↑</td><td>DetA↑</td><td>IDF1↑</td><td>MOTA↑</td><td>IDs↓</td><td>MDA↑</td><td>CVIDP↑</td><td>CVIDF1↑</td><td>CVMA↑</td></tr><tr><td>Base</td><td>=</td><td>=</td><td>一</td><td>一</td><td>=</td><td>.6240</td><td>.5548</td><td>.7999</td><td>.6801</td><td>19</td><td>.3995</td><td>.5602</td><td>.4881</td><td>.5395</td></tr><tr><td>+ Reliability Controller</td><td>√</td><td>1</td><td>1</td><td>=</td><td></td><td>.6386</td><td>.5802</td><td>.8246</td><td>.6989</td><td>46</td><td>.4769</td><td>.6271</td><td>.5608</td><td>.5805</td></tr><tr><td>+ Cross-view Assistance</td><td>√</td><td>√</td><td>=</td><td>=</td><td>=</td><td>.6389</td><td>.5890</td><td>.8280</td><td>.7100</td><td>45</td><td>.5360</td><td>.6460</td><td>.6000</td><td>.6250</td></tr><tr><td>+ Online Contrastive</td><td>√</td><td>√</td><td>√</td><td>=</td><td>一</td><td>.6528</td><td>.5964</td><td>.8453</td><td>.7322</td><td>83</td><td>.5255</td><td>.6271</td><td>.5850</td><td>.6398</td></tr><tr><td>Full w/o Rel.</td><td>I</td><td>√</td><td>√</td><td>√</td><td>√</td><td>.6178</td><td>.5482</td><td>.7938</td><td>.6710</td><td>32</td><td>.3924</td><td>.5507</td><td>.4796</td><td>.5333</td></tr><tr><td>w/o Online</td><td>√</td><td>√</td><td>一</td><td>√</td><td>√</td><td>.6250</td><td>.5650</td><td>.8050</td><td>.6900</td><td>26</td><td>.5140</td><td>.6210</td><td>.5710</td><td>.6210</td></tr><tr><td>w/o Verified Updates</td><td>L</td><td>V</td><td>√</td><td>V</td><td>I</td><td>.6390</td><td>.5880</td><td>.8240</td><td>.7090</td><td>57</td><td>.5190</td><td>.6220</td><td>.5830</td><td>.6410</td></tr><tr><td>Adaptive Query</td><td>√</td><td>7</td><td>√</td><td>√</td><td>√</td><td>.6370</td><td>.5810</td><td>.8250</td><td>.7110</td><td>44</td><td>.5330</td><td>.6410</td><td>.5960</td><td>.6510</td></tr><tr><td>Ours (Full)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>.6438</td><td>.5861</td><td>.8333</td><td>.7217</td><td>28</td><td>.5503</td><td>.6576</td><td>.6129</td><td>.6677</td></tr></table>

Metrics. Single-view over-time identity maintenance is evaluated by aggregating results independently computed in each view, including HOTA, DetA, AssA, IDF1, MOTA, MOTP, and identity switches (IDs) Bernardin & Stiefelhagen (2008); Ristani et al. (2016); Luiten et al. (2021). IDs denotes TrackEval’s standard IDSW count. Cross-view identity association is evaluated using CVIDP, CVIDR, CVIDF1, MDA, and CVMA Gan et al. (2021); Fan et al. (2025). Initialization fixes the selected targets and output labels; metric matching follows the evaluation rules in Appendix B.2.

Implementation. We use YOLOX Ge et al. (2021) and BoT-SORT Aharon et al. (2022) to generate frame-level observations for each view. Following OCLReID Ye et al. (2024), observations are encoded by a ResNet-18 based ReID backbone. Association, memory admission, and adapter fitting are detailed in Appendix A, with numerical settings in Appendix B.1.

## 4.2 OVERALL RESULTS

M3Track. Table 1 reports the single-view and cross-view metrics on the M3Track Fan et al. (2025) benchmark. Compared with online baselines such as OCLReID, BoT-SORT, ByteTrack, and OC-SORT+RPF, our method achieves higher IDF1 and CVIDF1. In particular, it increases IDF1 from .7424 to .8333 and CVIDF1 from .3918 to .6129 over OCLReID, the strongest online baseline on both metrics. These gains indicate improved identity continuity within views and consistency across views under the reported evaluation measures. We further compare with offline-trained MVMT references. Relative to CrossMOT, our method improves IDF1 from .7945 to .8333 and CVIDF1 from .5786 to .6129, while its MDA is lower (.5503 versus .5635). AssA, MOTP, and IDs are not the best in the table. These results show competitive performance against offline references, whose additional target-selection information is detailed in Appendix B.2.

MvMHAT. Table 2 reports the comparison with online and offline methods on MvMHAT. The gains over online baselines extend to this benchmark: IDF1 increases from .5292 for OCLReID to .7323, while CVIDF1 increases from .3068 for ByteTrack to .5685. These are the strongest online baselines for the metrics. Compared with CrossMOT, our method achieves higher IDF1 (.7323 versus .6901), but lower CVIDF1 (.5685 versus .6078). Its cross-view metrics remain below CrossMOT, and it incurs more identity switches. Together, the benchmarks show IDF1 and CVIDF1 gains over the online baselines, while the comparison with offline references remains mixed.

## 4.3 ABLATION STUDY

Table 3 examines component combinations and full-model controls (Appendix B.3). Reliability control improves IDF1 and CVIDF1, while cross-view assistance raises CVIDF1 from .5608 to .6000. Adding online contrastive adaptation improves HOTA and IDF1 but lowers CVIDF1 and increase IDs. Full adds Gaussian verification with its write and learning constraints, raising CVIDF1 to .6129 and reducing IDs from 83 to 28, at the cost of lower IDF1 than the Online Contrastive variant.

The controls show that online adapter learning remains useful within the system. Disabling adapter optimization while retaining Ridge refitting lowers IDF1 and CVIDF1 by 2.83 and 4.19 points. With adapter learning retained, removing verification-gated updates lowers CVIDF1 by 2.99 points and

Predicted target

![](images/4d51bbc654b020affc669ebc26997d8358541591feb2b745becd817e12fcebb5.jpg)  
Target (GT)  
Figure 3: Identity maintenance in the M3Track garden scene. Columns show synchronized observations; the first row shows the reference view. Yellow dashed boxes mark the target, and cyan boxes mark confirmed predictions.

increases IDs from 28 to 57. These comparisons support learning from new observations while restricting which observations may update persistent memory and supervise adaptation.

Using adapted rather than canonical verification queries lowers CVIDF1 by 1.69 points and increases IDs from 28 to 44. This supports retaining canonical verification queries while adapting representations for association. Because Gaussian statistics remain canonical in this control, the comparison addresses query choice under canonical statistics, not the necessity of the entire two-space design.

## 4.4 QUALITATIVE ANALYSIS

Figure 3 shows an identity maintenance example on M3Track. As the target moves near other pedestrians, the configuration without cross-view assistance assigns the target identity to another person wearing a similar white shirt. This incorrect association persists through the displayed observations, while the selected target remains visible. With cross-view assistance, the controller maintains the selected identity through the same interval. The other view provides complementary observations of the target, and the recovery step records the use of cross-view memory. This example illustrates cross-view assistance, not the isolated effects of long-term verification or online adaptation. Appendix B.4 compares incorrect assignments and interrupted outputs in the wood scene.

## 5 CONCLUSION

In this paper, we study online multi-view identity maintenance, where targets are specified once and tracked across views without further identity supervision. To address this problem, we propose OMNIGLIMPSE, which uses asymmetric memory roles to separate association proposals from verified updates, and asymmetric representation spaces to adapt appearance features while keeping identity verification in a stable shared space. Together, these choices support tracking and adaptation across views with only initialization supervision, without scene-specific training or camera calibration. Experiments on M3Track and MvMHAT show stronger identity maintenance than the evaluated online baselines and competitive performance against offline-trained references. The current evaluation focuses on a small set of designated targets with per-view initialization; scaling the framework to denser target sets and weaker initialization remains an important direction. We will release the code to support reproducibility and future research. We hope OmniGlimpse will advance multi-view tracking toward sustained identity maintenance in changing environments.

## AI USE STATEMENT

Generative AI tools were used to assist with language polishing and to support literature retrieval and discovery during the preparation of this manuscript. All retrieved materials were independently checked by the authors, and all AI-assisted text was reviewed and revised to ensure that it accurately reflects the authors’ intended meaning and the reported work. The authors take full responsibility for the final manuscript, including its technical content, claims, citations, and conclusions.

## REPRODUCIBILITY STATEMENT

Implementation details are provided in Appendix A, with model configurations and online-update settings in Appendix B.1. Evaluation protocols and baseline adaptations are described in Appendix B.2, and ablation controls are specified in Appendix B.3. We will release the code and experiment configurations to facilitate reproducibility.

## REFERENCES

Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multi-pedestrian tracking. arXiv preprint arXiv:2206.14651, 2022.

Temitope Ibrahim Amosa, Patrick Sebastian, Lila Iznita Izhar, Oladimeji Ibrahim, Lukman Shehu Ayinla, Abdulrahman Abdullah Bahashwan, Abubakar Bala, and Yau Alhaji Samaila. Multicamera multi-object tracking: A review of current trends and future advances. Neurocomputing, 552:126558, 2023.

Gorkay Aydemir, Xiongyi Cai, Weidi Xie, and Fatma Guney. Track-on: Transformer-based online¨ point tracking with memory. In International Conference on Learning Representations, volume 2025, pp. 57112–57134, 2025.

Keni Bernardin and Rainer Stiefelhagen. Evaluating multiple object tracking performance: the clear mot metrics. EURASIP Journal on Image and Video Processing, 2008(1):246309, 2008.

Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric sort: Rethinking sort for robust multi-object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9686–9696, 2023.

Cheng-Che Cheng, Min-Xuan Qiu, Chen-Kuo Chiang, and Shang-Hong Lai. Rest: A reconfigurable spatial-temporal graph model for multi-camera multi-object tracking. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10051–10060, 2023a.

De Cheng, Xiaolong Wang, Nannan Wang, Zhen Wang, Xiaoyu Wang, and Xinbo Gao. Crossmodality person re-identification with memory-based contrastive embedding. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 425–432, 2023b. doi: 10.1609/ aaai.v37i1.25116.

Yutao Cui, Chenkai Zeng, Xiaoyu Zhao, Yichun Yang, Gangshan Wu, and Limin Wang. Sportsmot: A large multi-object tracking dataset in multiple sports scenes. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9921–9931, 2023.

Zuozhuo Dai, Guangyuan Wang, Weihao Yuan, Siyu Zhu, and Ping Tan. Cluster contrast for unsupervised person re-identification. In Proceedings of the Asian conference on computer vision, pp. 1142–1160, 2022.

Shuxiao Ding, Lukas Schneider, Marius Cordts, and Juergen Gall. Ada-track: End-to-end multicamera 3d multi-object tracking with alternating detection and association. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15184–15194, 2024.

Huijie Fan, Yu Qiao, Yihao Zhen, Tinghui Zhao, Baojie Fan, and Qiang Wang. All-day multicamera multi-target tracking. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 16892–16901, 2025.

Yiyang Gan, Ruize Han, Liqiang Yin, Wei Feng, and Song Wang. Self-supervised multi-view multihuman association and tracking. In Proceedings of the 29th ACM international conference on multimedia, pp. 282–290, 2021.

Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. Yolox: Exceeding yolo series in 2021. arXiv preprint arXiv:2107.08430, 2021.

Shengyu Hao, Peiyuan Liu, Yibing Zhan, Kaixun Jin, Zuozhu Liu, Mingli Song, Jenq-Neng Hwang, and Gaoang Wang. Divotrack: A novel dataset and baseline method for cross-view multi-object tracking in diverse open scenes. International Journal of Computer Vision, 132(4):1075–1090, 2024.

Hsiang-Wei Huang, Cheng-Yen Yang, Zhongyu Jiang, Pyong-Kun Kim, Kyoungoh Lee, Kwangju Kim, Samartha Ramkumar, Chaitanya Mullapudi, In-Su Jang, Chung-I Huang, et al. Enhancing multi-camera people tracking with anchor-guided clustering and spatio-temporal consistency id re-assignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5239–5249, 2023.

Adilbek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, and Eric Xing. Efficient test-time adaptation of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14162–14171, 2024.

Lihao Liu, Yanqi Cheng, Zhongying Deng, Shujun Wang, Dongdong Chen, Xiaowei Hu, Pietro Lio,\` Carola-Bibiane Schonlieb, and Angelica Aviles-Rivero. Trafficmot: a challenging dataset for¨ multi-object tracking in complex traffic scenarios. In Proceedings ofthe 32nd ACM International Conference on Multimedia, pp. 1265–1273, 2024.

Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip Torr, Andreas Geiger, Laura Leal-Taixe,´ and Bastian Leibe. Hota: A higher order metric for evaluating multi-object tracking. International journal ofcomputer vision, 129(2):548–578, 2021.

Gerard Maggiolino, Adnan Ahmad, Jinkun Cao, and Kris Kitani. Deep OC-SORT: Multi-pedestrian tracking by adaptive re-identification. In 2023 IEEE International Conference on Image Processing (ICIP), pp. 3025–3029. IEEE, 2023.

Ergys Ristani, Francesco Solera, Roger Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In European conference on computer vision, pp. 17–35. Springer, 2016.

Mattia Segu, Bernt Schiele, and Fisher Yu. DARTH: Holistic test-time adaptation for multiple object tracking. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9683– 9693. IEEE, 2023.

Kyujin Shim, Kangwook Ko, Yujin Yang, and Changick Kim. Focusing on tracks for online multiobject tracking. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11687–11696. IEEE, 2025.

Andreas Specker. Ocmctrack: Online multi-target multi-camera tracking with corrective matching cascade. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7236–7244, 2024.

Jovana Videnovic, Matej Kristan, and Alan Luke´ ziˇ c. Distractor-aware memory-based visual objectˇ tracking. International Journal ofComputer Vision, 134(5):211, 2026.

Sanghyun Woo, Kwanyong Park, Inkyu Shin, Myungchul Kim, and In So Kweon. Mtmmc: a large-scale real-world multi-modal camera tracking benchmark. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22335–22346, 2024.

Zhenyu Xie, Zelin Ni, Wenjie Yang, Yuang Zhang, Yihang Chen, Yang Zhang, and Xiao Ma. A robust online multi-camera people tracking system with geometric consistency and state-aware re-id correction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7007–7016, 2024.

Taiga Yamane, Ryo Masumura, Satoshi Suzuki, and Shota Orihashi. Mvtrajecter: Multi-view pedestrian tracking with trajectory motion cost and trajectory appearance cost. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13270–13280, 2025.

H. Ye, J. Zhao, Y. Zhan, W. Chen, L. He, and H. Zhang. Person re-identification for robot person following with online continual learning. IEEE Robotics and Automation Letters, 2024.

Hanjing Ye, Jieting Zhao, Yaling Pan, Weinan Chen, and Hong Zhang. Following closely: A robust monocular person following system for mobile robot. arXiv preprint arXiv:2204.10540, 2022.

Mang Ye, Jianbing Shen, Gaojie Lin, Tao Xiang, Ling Shao, and Steven CH Hoi. Deep learning for person re-identification: A survey and outlook. IEEE transactions on pattern analysis and machine intelligence, 44(6):2872–2893, 2021.

Ram Zaveri, Shivang Patel, Yu Gu, and Gianfranco Doretto. Improving accuracy and generalization for efficient visual tracking. In Proceedings ofthe Winter Conference on Applications ofComputer Vision, pp. 9450–9460, 2025.

Tianyuan Zhang, Xuanyao Chen, Yue Wang, Yilun Wang, and Hang Zhao. Mutr3d: A multicamera tracking framework via 3d-to-2d queries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4537–4546, 2022a.

Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In European conference on computer vision, pp. 1–21. Springer, 2022b.

Jinglin Zhou, Tianyang Xu, Xuefeng Zhu, Xiao-Jun Wu, and Josef Kittler. SPD-Updater: Symmetric positive definite manifold geometry based temporal updating for visual object tracking. Neural Networks, 193:107985, 2026. doi: 10.1016/j.neunet.2025.107985.

## A METHOD DETAILS

## A.1 CANDIDATE ASSOCIATION DETAILS

Prior identity reference. While the classifiers adapt online, initialization provides an identity reference before pseudo-labels are available. For target m in view v, let $\mathcal { P } _ { m , v } ^ { \mathrm { i n i t } }$ contain the crops supplied at that view’s initialization. A reference is constructed for each available cue:

$$
r _ { k } ^ { ( m , v ) } = \frac { 1 } { | \mathcal { P } _ { m , v , k } ^ { \mathrm { i n i t } } | } \sum _ { p \in \mathcal { P } _ { m , v , k } ^ { \mathrm { i n i t } } } g _ { k } ( p ) , \qquad k \in \{ 0 , \ldots , K \} ,\tag{10}
$$

where $\mathcal { P } _ { m , v , k } ^ { \mathrm { i n i t } }$ includes only initialization crops in which cue k is available. The global cue is always available. These canonical references remain fixed after the corresponding initialization; reference crops from another view can enter the support bank only after that view’s initialization has occurred and its message has arrived. The prior score combines global cosine similarity with the mean cosine similarity over jointly available parts, using weights 0.7 and 0.3, respectively. When no part is jointly available, only the global similarity is used. The resulting score is clipped to [0, 1].

Orientation-aware classifiers. Each target–view unit maintains orientation-aware ridge classifiers $\{ c _ { \omega , k } ^ { ( m , v ) } \}$ , where $\omega \in \{ \mathrm { f r o n t } , \mathrm { b a c k } \}$ and $k = 0$ denotes the global cue. Each classifier is trained as a binary target/non-target regressor with labels one and zero. For observation $^ { O , }$ only trained classifiers matching its predicted orientation and visible cues are evaluated. Let $\mathcal { K } _ { t } ^ { ( m , v ) } ( o )$ denote this set. Their mean response is

$$
C _ { t } ^ { ( m , v ) } ( o ) = \frac { 1 } { \vert \mathcal { K } _ { t } ^ { ( m , v ) } ( o ) \vert } \sum _ { k \in \mathcal { K } _ { t } ^ { ( m , v ) } ( o ) } c _ { \omega ( o ) , k } ^ { ( m , v ) } ( \widehat { z } _ { t , k } ^ { ( m , v ) } ( o ) ) .\tag{11}
$$

The local association score is $\mathrm { c l i p } ( C _ { t } ^ { ( m , v ) } ( o ) , 0 , 1 )$ . If no applicable classifier is trained, the prior score is used instead. This clipping gives a bounded ranking score, not a calibrated posterior probability. All target-specific classifiers use the same feature-normalization convention, ridge regularization, and equal total weights for positive and negative training samples. Fusion weights and acceptance thresholds are shared across targets rather than fitted independently to each target. These common settings constrain score construction but do not guarantee cross-target calibration. The classifier buffer stores canonical features and their admitted labels; it does not treat every emitted association as a training sample. Refitting uses features mapped by the current local adapter.

For each orientation/cue classifier, let $B _ { + }$ and B contain admitted positive and negative samples with the required orientation and visible cue. Negatives carry an admitted identity different from the target, as specified in Appendix A.3. With $x _ { i }$ the normalized feature mapped through the current target-specific adapter, fitting solves

$$
\operatorname* { m i n } _ { w , b } \ \frac { 1 } { 2 | \boldsymbol { \mathcal { B } } _ { + } | } \sum _ { i \in \boldsymbol { \mathcal { B } } _ { + } } ( w ^ { \top } \boldsymbol { x } _ { i } + b - 1 ) ^ { 2 } + \frac { 1 } { 2 | \boldsymbol { \mathcal { B } } _ { - } | } \sum _ { i \in \boldsymbol { \mathcal { B } } _ { - } } ( w ^ { \top } \boldsymbol { x } _ { i } + b ) ^ { 2 } + \lambda _ { R } \| w \| _ { 2 } ^ { 2 } , \qquad \lambda _ { R } > 0 .\tag{12}
$$

The intercept is unpenalized, and each class receives half the total sample weight. The response is $c ( x ) = w ^ { \top } x + b .$ . A classifier is fitted only when both sets are nonempty. Otherwise it is disabled and excluded from $\kappa _ { t } ^ { ( m , v ) } ( o )$ ; if that set is empty, scoring falls back to the prior. Initialization labels can supply either class, but bootstrap pseudo-labels cannot. A single-target deployment without admitted negatives therefore uses prior and memory scores rather than a one-class Ridge fit.

An affected classifier is refit whenever its admitted buffer changes, including insertion, eviction, or a change in training eligibility, even if the adapter does not change. An adapter update invalidates every affected orientation/cue model: its training features are remapped from canonical storage and it is refit before the next association step. Missing either class disables the model rather than retaining stale weights. If neither its buffer nor its adapter changes, the fitted model is reused. Newly available initialization labels are incorporated before scoring; other admitted samples affect subsequent steps only.

Support bank and temporal eligibility. The support bank combines recent local target observations, received target observations from other views, and available initialization references. Each sample retains its source view, timestamp, local track identity, canonical global/part features, and visibility mask. Remote samples are eligible only if their timestamps do not exceed the current processing time and their age is within a synchronization window $\Delta _ { \mathrm { s y n c } } .$ . Features and masks are inserted and removed together. A missing part mask makes the sample unavailable for part comparison, but does not prevent global comparison. Tentative short-term support may help propose an association without being admitted for representation learning.

Global and part support. For brevity, suppress $( m , v , t )$ and let $z _ { k } ( o )$ and $z _ { k } ( b )$ denote the query and support features produced by the same current local adapter. Global pair similarity is $h _ { g } ( o , b ) = \cos ( z _ { 0 } ( o ) , z _ { 0 } ( b ) )$ ). For each support sample $b ,$ define the jointly visible parts $\mathcal { \bar { I } } ( o , b ) \ \stackrel {  } { = } \{ k \in \{ 1 , \dotsc , K \}  : w _ { k } ( \omega ) > 0 , w _ { k } ( b ) > \bar { 0 } \}$ and

$$
h _ { \boldsymbol { p } } ( \boldsymbol { o } , \boldsymbol { b } ) = \frac { 1 } { | \mathcal { I } ( \boldsymbol { o } , \boldsymbol { b } ) | } \sum _ { \boldsymbol { k } \in \mathcal { I } ( \boldsymbol { o } , \boldsymbol { b } ) } \cos \bigl ( z _ { \boldsymbol { k } } ( \boldsymbol { o } ) , z _ { \boldsymbol { k } } ( \boldsymbol { b } ) \bigr ) .\tag{13}
$$

Samples with $\mathcal { I } ( o , b ) = \emptyset$ are excluded from part aggregation. The global and part support scores, $G ( o )$ and $P ( o )$ , are the means of the largest L valid pair similarities in their respective channels. Fewer than L samples use all available values; an empty channel returns zero.

Part scores are used where available when a discriminativeness criterion is met: among all current observations for the target, at least two have positive part support, the largest part score is at least $\tau _ { p } ,$ , and its margin over the second largest is at least $\delta _ { p }$ . For candidates with positive part support under this condition, $s _ { \mathrm { m e m } } ( o ) = \mathrm { c l i p } ( P ( o ) , 0 , 1 )$ ; otherwise it is $\operatorname { c l i p } ( G ( o ) , 0 , 1 )$ . Thus missing or ambiguous part evidence falls back to global support. An empty support bank disables memory assistance rather than treating a zero memory score as negative evidence.

Binding retention and conflict handling. A current binding is eligible for retention when its observation remains present and its local score exceeds a tracking threshold $\tau _ { \mathrm { k e e p } }$ . It is retained only if no other target’s existing binding meets the same retention condition for that observation. Such contested bindings enter reassignment rather than being resolved by target iteration order; a newly scored candidate alone does not count as a competing retained binding. Targets without a retained

binding activate memory assistance. Candidate pairs must exceed an association threshold τ<sub>assoc</sub>.   
When entering recovery after losing a binding, they must also satisfy a prior-support threshold τ<sub>prior</sub>.   
A target’s verified initialization observation takes precedence when initialization occurs.

Eligible pairs are sorted by association score. Each pair is selected only if both its target and observation remain unassigned; ties are resolved by target index and then local track index. After allocation, pairs whose boxes exceed an overlap threshold $\tau _ { \mathrm { I o U } }$ cannot both remain: verified initialization pairs have priority, followed by retained bindings and then association score. A suppressed pair does not trigger another search in the same frame. These operations generate provisional associations, not permission to train on the selected observations. They do not impose universal targetand track-margin rejection thresholds.

## A.2 LONG-TERM DISTRIBUTION DETAILS

Initialization and bootstrap. Each local distribution is seeded with the normalized global features of initialization crops available at that view. New prototypes start with diagonal variance $\sigma _ { \mathrm { i n i } } ^ { 2 }$ t within the clipping interval $[ \sigma _ { \mathrm { m i n } } ^ { 2 } , \sigma _ { \mathrm { m a x } } ^ { 2 } ]$ Before the total count reaches $N _ { \mathrm { m i n } }$ , lack of a mature distribution does not veto an association. Bootstrap observations must instead satisfy prior support, association confidence, competing-target and competing-observation margins, and consecutive binding stability. They may populate the long-term distribution but are not treated as verified adaptation samples. Thus, bootstrap admission establishes the reference needed for verification without authorizing ordinary pseudo-label training before that reference is mature.

Admission conditions. Both margins are computed from a pre-allocation score matrix $Q _ { t } ^ { ( v ) } ( m , i ) \ = \ s _ { \mathrm { a s s o c } } ^ { ( m , v ) } ( o _ { t . i } ^ { ( v ) } )$ . Its rows include all targets already initialized in view $v ,$ including targets with retained bindings, and its columns include all current observations. Retention status and memory-assistance weights are determined first; all pair scores are then computed before acceptance filtering, greedy allocation, overlap suppression, and long-term verification. For a target m and observation $o _ { i }$ , the margins are

$$
\begin{array} { l } { { \Delta _ { \mathrm { t g t } } ( m , i ) = Q _ { t } ^ { ( v ) } ( m , i ) - \displaystyle \operatorname* { m a x } _ { m ^ { \prime } \neq m } Q _ { t } ^ { ( v ) } ( m ^ { \prime } , i ) , } } \\ { { \Delta _ { \mathrm { o b s } } ( m , i ) = Q _ { t } ^ { ( v ) } ( m , i ) - \displaystyle \operatorname* { m a x } _ { i ^ { \prime } \neq i } Q _ { t } ^ { ( v ) } ( m , i ^ { \prime } ) . } } \end{array}\tag{14}
$$

If a competitor set is empty, that margin imposes no restriction. The matrix is not reduced after assignment, so selecting one target–observation pair does not erase its competitors from the admission test. Long-memory admission requires association confidence at least $\tau _ { a } .$ , both margins at least $\tau _ { \Delta }$ and the same target–track binding for at least $\tau _ { T }$ consecutive observed frames. Missing observations or a changed binding reset this count. Bootstrap additionally requires prior support at least $\tau _ { \mathrm { b o o t } } ;$ after maturity, admission instead requires passing long-term verification. These are write conditions, not universal requirements for emitting a current association.

Prototype updates. An admitted canonical global feature x is assigned to the prototype with maximum cosine similarity. If this similarity is below $\tau _ { \mathrm { n e w } }$ and capacity remains, a new prototype is initialized with mean $x ,$ variance $\sigma _ { \mathrm { i n i t } } ^ { 2 }$ , and count one. Otherwise the nearest prototype is updated without replacing another mode. After bank maturity, a new prototype can be created only from evidence that already passes the existing verifier. Maturity is assessed at bank level: a newly created prototype participates in matching even before its individual count reaches $N _ { \mathrm { m i n } }$ . This permits additional modes within the accepted region but does not ensure recovery from abrupt appearance changes outside it. Let $\mu _ { \mathrm { o l d } }$ and $\bar { \sigma } _ { \mathrm { o l d } } ^ { 2 }$ be its parameters before the update and $\Delta = x - \mu _ { \mathrm { o l d } }$ . The update is

$$
\begin{array} { r l } & { \mu _ { \mathrm { n e w } } = ( 1 - \alpha ) \mu _ { \mathrm { o l d } } + \alpha x , } \\ & { \sigma _ { \mathrm { n e w } } ^ { 2 } = \mathrm { c l i p } _ { [ \sigma _ { \mathrm { m i n } } ^ { 2 } , \sigma _ { \mathrm { m a x } } ^ { 2 } ] } \left( ( 1 - \alpha ) \sigma _ { \mathrm { o l d } } ^ { 2 } + \alpha ( 1 - \alpha ) \Delta ^ { \odot 2 } \right) . } \end{array}\tag{15}
$$

The factor $( 1 - \alpha )$ in the residual term accounts for the moving mean. The prototype count is incremented by one for each newly admitted sample; it records accumulated support rather than the effective sample size of the exponential moving average. Variance clipping prevents degenerate uncertainty estimates. Matching uses minimum Mahalanobis distance, whereas update routing uses maximum cosine similarity.

Cross-view evidence. Each process maintains its own target distributions. A received longmemory event contains the canonical feature, source view, timestamp, and an observation identifier for deduplication. Sender-side admission does not automatically authorize a receiver-side update. At a mature receiver, an event may update the distribution only when the receiver has a local association for the same target at the current step that satisfies its own verification, confidence, margin, and stability conditions. The received feature must independently pass the receiver’s pre-update verifier and meet the temporal eligibility rule in Appendix A.1. Stability and margins are measured on the receiver’s local association, not inferred from the remote track identifier. Before maturity, the receiver instead requires a local association satisfying its bootstrap conditions, and the received feature must independently satisfy the bootstrap prior-support threshold in the canonical space. All checks use the receiver’s statistics at the start of the step. Without qualifying local evidence, remote samples can support candidate association but cannot update the receiver’s long-term distribution. Thus, cross-view evidence cannot directly mature that distribution while the target is unobserved locally. Repeated events are not counted twice. Local distributions can consequently differ while remaining comparable in the common canonical space.

Verification outcomes. A mature bank returns pass or fail according to the consistency threshold; an immature bank returns unavailable. Failed new or recovered proposals are removed without searching for a replacement in the same frame. A retained binding can remain visible in the output despite a failed check, but this does not admit its observation to long memory, Ridge training, or adapter training. An unavailable result does not by itself authorize learning; initialization labels and the restricted bootstrap procedure provide the starting evidence. All checks use pre-update statistics, so an observation cannot first change the reference against which it is verified.

## A.3 ONLINE ADAPTATION DETAILS

Residual architecture. For each target–view pair and cue $k ,$ the residual branch has two linear layers with a ReLU nonlinearity:

$$
h _ { \phi _ { k } } ( g ) = W _ { 2 , k } \mathrm { R e L U } ( W _ { 1 , k } g + b _ { 1 , k } ) + b _ { 2 , k } .\tag{16}
$$

The input and output dimensions equal the canonical cue dimension $D _ { k }$ , and the hidden width is $\operatorname* { m a x } ( \hat { 1 } , \lfloor D _ { k } / 4 \rfloor )$ . The first layer uses a standard variance-scaled initialization; the second-layer weights and bias are zero-initialized. Thus the residual output is initially zero without making both layers identically zero. The output is normalized before similarity computation, classifier fitting, and distribution anchoring. The canonical encoder, visibility estimator, and orientation predictor remain frozen. Adapters are local to each target–view pair and their parameters are not exchanged.

Sample permissions and negatives. Initialization observations carry their supplied labels. Other positive samples require admission under Sec. 3.3, including receiver-side checks for cross-view samples. Training eligibility is stored separately from presence in a short-term proposal bank. A negative for target m must carry an admitted identity $m ^ { \prime } \ne m ;$ not being selected for m is insufficient. Observations whose identity remains unresolved after conflict or overlap handling are excluded. Among eligible negatives, samples most similar to the anchor under the current target-m adapter are selected for hard-negative training. This procedure does not treat an arbitrary co-present detection as a negative. When no eligible negative is available, the contrastive term is skipped rathe than assigning an unverified negative label.

Cue aggregation and triplet construction. The positive pool combines training-eligible observations in the short-term support bank with the Ridge positive buffer, deduplicated by source view and observation identifier. Tentative support without training permission is excluded. Within each same-view or cross-view allocation, valid anchor–positive pairs are sampled uniformly without replacement, subject to the time-separation requirement. An observation cannot serve as both anchor and positive in a pair. Negatives follow the admitted hard-negative rule above.

All features in a triplet are recomputed from their cached canonical features through the same local adapter. The cue set is

$$
{ \cal K } ( a , p , n ) = \{ 0 \} \cup \{ k \in \{ 1 , \ldots , K \} : w _ { k } ( a ) > 0 , \ w _ { k } ( p ) > 0 , \ w _ { k } ( n ) > 0 \} .\tag{17}
$$

Missing part masks exclude the corresponding parts. The triplet loss averages cue-level losses:

$$
\ell _ { \mathrm { t r i } } ( a , p , n ) = \frac { 1 } { | \mathcal { K } ( a , p , n ) | } \sum _ { k \in \mathcal { K } ( a , p , n ) } \ell _ { \mathrm { t r i } , k } ( a , p , n ) .\tag{18}
$$

For the receiver view $v ,$ same-view triplets use a local anchor and a distinct local same-target positive; cross-view triplets use a local anchor and an admitted same-target positive from another view. Negatives may come from either source if their receiver-side training permission is valid. The losses are

$$
{ \mathcal { L } } _ { \mathrm { s v } } = { \frac { 1 } { | { \mathcal { T } } _ { \mathrm { s v } } | } } \sum _ { T _ { \mathrm { s v } } } \ell _ { \mathrm { t r i } } ( a , p , n ) , \qquad { \mathcal { L } } _ { \mathrm { c v } } = { \frac { 1 } { | { \mathcal { T } } _ { \mathrm { c v } } | } } \sum _ { T _ { \mathrm { c v } } } \ell _ { \mathrm { t r i } } ( a , p , n ) .\tag{19}
$$

An empty triplet set contributes zero. The two sets are averaged independently before applying γ<sub>cv</sub>, so sample counts do not implicitly determine the relative loss weight.

Anchor eligibility and update order. The set $\mathcal { P } _ { + }$ comprises distinct admitted target observations used as anchors or positives in the batch; negative observations are never pulled toward the target’s Gaussian distribution. If no valid triplet exists, no contrastive or anchor-only adapter step is performed. Adapter optimization begins only when the receiver’s target distribution is mature. The distribution snapshot taken before the current step’s memory writes is held fixed throughout all optimizer steps for that batch. The optimizer differentiates the anchor loss through the adaptive global feature but not through Gaussian statistics or the canonical encoder. Part features receive the contrastive loss without a separate Gaussian penalty.

After optimizing the adapter, the receiver invalidates adaptive-feature caches and recomputes them from stored canonical features. Orientation/cue Ridge classifiers with both admitted classes are refit in the new adaptive space using Appendix A.1; missing-class models are disabled, and old weights are never used with new features. Buffer changes also trigger refitting without an adapter update. Available classifiers are aggregated; if none applies, scoring uses the prior fallback. This refresh is completed before the next association step. No backbone forward pass on historical images and no synchronization of adapter parameters are required.

## B EXPERIMENTAL DETAILS

## B.1 IMPLEMENTATION SETTINGS

The canonical encoder uses the trained ResNet-18 checkpoint from OCLReID Ye et al. (2024).

Tables 4 and 5 summarize the configuration of the complete model. Ablation-specific changes are described in Appendix B.3. The canonical encoder and its feature output path, visibility estimator, and orientation predictor remain frozen; only the local residual adapters are optimized. Their architecture and initialization are specified above. Offline references are trained using two NVIDIA RTX A6000 GPUs, while evaluation uses a single RTX A6000 GPU.

Buffer and update conventions. Initialization references are retained separately from the bounded short support bank. Ridge buffers use FIFO eviction and retain the visibility information of each observation. Their capacity is per class, not per cue; each orientation/cue classifier uses the applicable subset. Contrastive negatives must have an admitted identity different from the target. Remote observations are used only after receipt, with timestamps no later than the current processing time and ages within $\Delta _ { \mathrm { s y n c } }$

The optimization interval counts newly admitted local observations with training permission, not elapsed video frames or received messages. Each eligible batch contains at least four valid triplets. The sampling proportions specify the intended allocation; unavailable triplets are not duplicated to fill a batch. The two loss groups are averaged separately as in Eq. 19. Long-memory writes and pseudo-label training use the same confidence, margin, and stability requirements after bank maturity. Bootstrap observations only establish the reference and do not count toward adaptertraining eligibility. Adapter updates use pre-write Gaussian statistics, followed by feature remapping and Ridge refitting before the next association step.

Table 4: Association, verification, and admission settings. Capacities count observations or prototypes, not individual feature coordinates.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Detector / canonical backbone</td><td>YOLOX-L / ResNet-18</td></tr><tr><td>Canonical checkpoint source</td><td>OCLReID-trained ResNet-18</td></tr><tr><td>Number of part cues  $K$ </td><td>4</td></tr><tr><td>Global and part dimensions  $D _ { k }$ </td><td>512</td></tr><tr><td>Detection score / NMS IoU threshold</td><td>0.60 / 0.35</td></tr><tr><td>Support aggregation L</td><td>3</td></tr><tr><td>Active memory fusion weight  $\lambda _ { \mathrm { m e m } }$ </td><td>0.80</td></tr><tr><td>Prior global / mean-part weights Part-support threshold  $\tau _ { p } \mathrm { ~ / ~ }$  margin  $\delta _ { p }$ </td><td>0.70 / 0.30</td></tr><tr><td>Binding retention</td><td>0.80 / 0.005</td></tr><tr><td> $\pi _ { \mathrm { k e e p } }$  Association acceptance</td><td>0.60</td></tr><tr><td> $\tau _ { \mathrm { a s s o c } }$ </td><td>0.70</td></tr><tr><td>Recovery prior support  $\tau _ { \mathrm { p r i o r } }$ </td><td>0.72</td></tr><tr><td>Assignment overlap threshold  $\scriptstyle \tau _ { \mathrm { I o U } }$ </td><td>0.70</td></tr><tr><td>Maximum remote-observation age  $\Delta _ { \mathrm { s y n c } }$ </td><td>0.10 s</td></tr><tr><td>Short support capacity per target and source view</td><td>64 observations</td></tr><tr><td>Gaussian capacity per target-view pair</td><td>4 prototypes</td></tr><tr><td>Prototype update rate α</td><td>0.05</td></tr><tr><td>Initial diagonal variance  $\sigma _ { \mathrm { i n i t } } ^ { 2 }$ </td><td>1/512</td></tr><tr><td>Variance bounds  $[ \sigma _ { \mathrm { m i n } } ^ { 2 } , \sigma _ { \mathrm { m a x } } ^ { 2 } ]$ </td><td> $[ \mathrm { { i } 0 ^ { - 3 } , 1 } ]$ </td></tr><tr><td>Distance denominator stabilizer €</td><td> $\mathrm { { i 0 ^ { - 6 } } }$ </td></tr><tr><td>New-prototype cosine threshold  $\tau _ { \mathrm { n e w } }$ </td><td></td></tr><tr><td>Bank maturity  $N _ { \mathrm { m i n } }$ </td><td>0.75</td></tr><tr><td>Verification score threshold  $\tau _ { \ell }$ </td><td>10 observations</td></tr><tr><td></td><td>0.50</td></tr><tr><td>Admission score  $\tau _ { a }$  / competing margins  $\tau _ { \Delta }$ </td><td>0.90 / 0.15</td></tr><tr><td>Consecutive binding stability τT</td><td>10 observed frames</td></tr><tr><td>Bootstrap prior-support threshold  $\scriptstyle { \mathcal { T } } _ { \mathrm { b o o t } }$ </td><td>0.80</td></tr></table>

Table 5: Local adapter learning and classifier settings. Ridge regularization refers to the classnormalized objective in Eq. 12.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Adapter hidden width</td><td>128</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Learning rate / weight decay</td><td> $1 0 ^ { - 4 } / 1 0 ^ { - 4 }$ </td></tr><tr><td>Optimization interval per target-view pair</td><td>5 newly admitted local observations</td></tr><tr><td>Optimizer steps per eligible batch</td><td>1</td></tr><tr><td>Maximum / minimum triplets per batch</td><td>16/4</td></tr><tr><td>Same-view / cross-view sampling proportion</td><td>0.5 / 0.5</td></tr><tr><td>Minimum same-view positive time separation</td><td>0.10 s</td></tr><tr><td>Triplet margin</td><td>0.20</td></tr><tr><td>Cross-view loss weight  $\gamma _ { \mathrm { c v } }$ </td><td>0.50</td></tr><tr><td>Distribution-anchor weight  $\beta _ { \mathrm { d i s t } }$ </td><td>0.004</td></tr><tr><td>Ridge regularization  $\lambda _ { R }$ </td><td>0.01</td></tr><tr><td>Ridge buffer per target-view pair and class</td><td>32 observations</td></tr><tr><td>Contrastive negative pool per target-view pair</td><td>256 observations</td></tr></table>

## B.2 EVALUATION PROTOCOL AND BASELINE ADAPTATION

Target selection and initialization. Each scene contains two designated target identities, with a fixed mapping to dataset identities shared across views and methods. M3Track reads this mapping from the target metadata; MvMHAT uses dataset identities 1 and 2. The four online baselines receive a bounding box at each target’s first annotated occurrence in each view’s test segment. Visible target frames remain in evaluation even if initialization fails. Only selected identities are retained in the evaluation ground truth; missed targets and false outputs remain subject to the evaluator’s matching rules.

Metric computation and aggregation. Tracking metrics are computed with TrackEval, treating each view as a separate sequence and retaining selected-target GT. The selected-target exporter uses the initialized target label, mapped to its dataset ID, as the prediction ID rather than the underlying local track ID. This fixes exported labels but does not constrain metric matching by label equality: CLEAR uses spatial matching with its temporal preference, and Identity uses global trajectory assignment. Dataset-level results combine sequence statistics rather than averaging scene scores; IDs are summed, and MOTP measures matched-box IoU. Cross-view scores are computed for synchronized view pairs, then averaged over scenes on M3Track and over all scene–view pairs on MvMHAT. Thus, four-view scenes contribute six pairs and three-view scenes three pairs. Both benchmark wrappers use the same pairwise evaluation function described below. These metrics assess tracking and association consistency rather than enforcing the initialized identity name at every match.

Identity switches. The IDs columns use the CLEAR IDSW field returned by TrackEval, also used in MOTA. An identity switch is counted when a GT identity is matched to a prediction ID different from its previous matched prediction ID. A gap does not by itself reset that previous ID. This is not a count of wrong-identity frames or the separate selected-target diagnostic switch counter.

Cross-view matching and empty frames. Within each view, each GT box is matched to the prediction with the highest IoU among candidates with IoU at least 0.5; if no such prediction exists, the GT is treated as unmatched. The selected prediction label is then used to construct the GT-toprediction identity map. A shared GT identity is cross-view correct when its matched prediction labels agree across the two views; equality to the GT identity label is not required.

Let $G _ { t }$ be the number of GT identities common to both views, $R _ { t }$ the number of distinct positive prediction labels shared by the two GT-to-prediction maps, and $C _ { t }$ the number of cross-view-correct GT identities. Let $Q _ { t }$ count GT entries in view a whose mapped prediction label occurs in both maps. Frame precision and recall are $C _ { t } / Q _ { t }$ and $C _ { t } / G _ { t } ,$ , respectively. Frame MDA is

$$
\frac { C _ { t } } { G _ { t } + \operatorname* { m a x } ( G _ { t } - C _ { t } , 0 ) + \operatorname* { m a x } ( R _ { t } - C _ { t } , 0 ) } .
$$

Zero-denominator values contribute zero. If either view has no GT, frame precision, recall, and MDA also contribute zero; such frames remain in the averaging denominator. With GT in both views but no predictions in one view, $C _ { t } = R _ { t } = Q _ { t } = 0 \quad$ , so these three frame scores are zero. CVIDP, CVIDR, and MDA are frame means; pair CVIDF1 is the harmonic mean of pair CVIDP and CVIDR before averaging across pairs.

For CVMA, let $E _ { t } = \operatorname* { m a x } ( R _ { t } - C _ { t } , 0 )$ and $M _ { t } = 2 \operatorname* { m a x } ( G _ { t } - R _ { t } , 0 )$ . Let $F _ { t }$ count distinct predicted labels present in both views but absent from one view’s GT-to-prediction map, summed over the two directions. With $N _ { t }$ the total number of GT boxes in both views,

$$
\mathrm { C V M A } = 1 - \frac { \sum _ { t } ( 2 E _ { t } + F _ { t } + M _ { t } ) } { \sum _ { t } N _ { t } } .
$$

When either view has no GT, $E _ { t } , F _ { t } , M _ { t }$ are zero, but available GT boxes still contribute to $N _ { t }$ . A zero total GT-box count gives CVMA zero.

Baseline adaptation. The online baselines run independently for each target and view, without exchanging observations, features, trajectories, or association decisions across cameras. Since these baselines do not perform cross-view association, we assign outputs for the same initialized target a

common label across views. This supplies cross-view label correspondence without requiring the baselines to infer it. Their cross-view scores are then computed from these aligned outputs using the same evaluation rules.

## B.3 FULL-MODEL ABLATION CONTROLS

The full-model controls retain the same observations, target initialization, association procedure, and non-ablated settings. They differ from the cumulative component rows because each changes one specified aspect of Full. Reliability control denotes the non-Gaussian admission conditions: the confidence threshold, both competing margins, and consecutive binding stability. It is distinct from V-Upd., which makes long-term verification a prerequisite for persistent writes and training. Base disables these additional reliability conditions, cross-view assistance, adapter optimization, and Gaussian verification; it retains local association and classifier fitting. Detection filtering, association acceptance, one-to-one allocation, and overlap handling remain common to all variants.

Without reliability control. Full w/o Rel. removes the confidence, competing-margin, and binding-stability conditions from ordinary memory and training admission. It retains Gaussian verification and verification-gated writes, cross-view assistance, and adapter learning. Initialization, restricted bootstrap, bank maturity, and valid-sample requirements remain unchanged. Thus, this control removes non-Gaussian reliability screening rather than long-term verification or basic association safeguards.

Cumulative online adaptation. For the cumulative Online Contrastive variant without Gauss., training admission uses the remaining reliability conditions without requiring long-term verification or a mature Gaussian bank, and the distribution-anchor loss is omitted. Thus, adding Gauss. changes both verification and its associated learning constraints.

Without online adaptation. The local adapters remain at their identity initialization, and no contrastive or distribution-anchor optimization is performed. Cross-view assistance, canonical-space verification, and memory admission remain active. Ridge classifiers are still refit when their admitted training buffers change, so this control removes representation learning rather than all online state updates.

Without verified updates. Long-term verification remains available for association decisions, but its pass/fail outcome no longer gates persistent-memory admission or training eligibility for the adapters and Ridge classifiers. This removal applies to both local and received observations. Confidence, competing margins, binding stability, and the requirement for a locally admitted same-target observation at the receiver are retained. Bootstrap and the requirement for a mature anchor before adapter learning are unchanged. This control tests verification-dependent write permissions rather than removing the entire reliability controller.

Adaptive verification features. This control replaces only the normalized canonical global query in long-term verification with its normalized output from the current local adapter; Gaussian statistics remain in canonical space, and their update rule, write permissions, and adaptation losses are unchanged. Both variants use the same verification threshold $\tau _ { \ell } = 0 . 5 0$ This comparison tests query substitution at a common threshold, not matched acceptance rates or the necessity of the entire two-space design.

Time

## B.4 ADDITIONAL QUALITATIVE ANALYSIS

Figure 4 shows an additional identity maintenance example in the M3Track wood scene. The selected target and a nearby pedestrian have similar dark clothing. The full model maintains the target identity throughout the displayed interval, whereas the configuration without online contrastive adaptation assigns it to the other pedestrian. Without cross-view assistance, the controller also selects the distractor before and after a brief interval with no confirmed output. Thus, the two ablations differ in whether an incorrect candidate is emitted, rather than exhibiting two consistently distinct failure modes.

The absence of a cyan box in the third column of the last row corresponds to this interruption. It lasts seven consecutive dataset frames (1868–1874), during which an incorrect tentative candidate remains available internally. The interruption therefore does not indicate that the identity has been corrected. These examples illustrate tracking behavior rather than isolate the effects of representation spaces or verification-dependent update permissions.

![](images/99acf79107c886f756c02a10d8a2e2b238f33047a931bc5e3b98261af1d7cf48.jpg)  
Target (GT)  
Figure 4: Additional comparison in the M3Track wood scene. Columns correspond to dataset frames 1861, 1867, 1871, and 1875. The reference view is shown once; the other rows use identical images and crops for the three configurations. Yellow dashed boxes denote the target and cyan boxes denote confirmed outputs; tentative candidates are not drawn. Displayed View 1 and View 2 correspond to source view2 and view1, respectively.