# Event-Driven Refresh and Recurrence Memory to Reduce Stale Grounding in Referring Video Object Segmentation

Abu Hanif Muhammad Syarubany, Jaehyun Jang, Siwoo Lim, Seungyeon Ryu, and Chang D. Yoo Korea Advanced Institute of Science & Technology (KAIST)

Abstract—Referring Video Object Segmentation (RVOS) aims to produce a pixel-accurate mask sequence for an object specified by natural language. Sa2VA combines a multimodal large language model with SAM2 for grounded segmentation; however, its inference typically grounds the query from a small fixed set of initial keyframes and then relies on propagation. In long or dynamic videos, this can cause stale grounding and persistent false positives when the object composition changes (e.g., distractors enter or the target disappears/re-appears). We propose Event-Driven Refresh + Recurrence Memory (EDRRM), an enhancement that selectively re-invokes Sa2VA only at stable change points. EDRRM triggers refresh boundaries using an EMA-smoothed event score computed from tracking-derived cues (births/deaths and coarse composition/layout changes) with temporal constraints. A recurrence memory further retrieves anchor frames via CLIP similarity to re-condition the model on re-appearance events. Experiments on Ref-DAVIS17, MeViS, and ReVOS show that EDRRM achieves a competitive accuracyefficiency trade-off relative to fixed-window and FrameDiff-SSIM baselines, maintaining comparable or superior J&F scores at substantially lower average refresh-call budgets and reducing false positive failures. End-to-end runtime analysis further confirms that the overhead introduced by tracking, CLIP-based recurrence matching, and the identifiability gate remains modest relative to the dominant Sa2VA inference cost, thereby validating the efficiency of the proposed pipeline.

Index Terms—Referring video object segmentation (RVOS), multimodal large language models (MLLMs), Segment Anything Model (SAM), SAM2, Sa2VA, temporal grounding, adaptive sampling

## I. INTRODUCTION

R <sup>EFERRING</sup> <sup>Video</sup> <sup>Object</sup> <sup>Segmentati</sup>to output a binary mask sequence to output a binary mask sequence $\{ \mathbf { M } _ { t } \} _ { t = 1 } ^ { T }$ S) aims<sub>for a</sub> for a target described by a natural-language query. Compared with conventional video object segmentation (VOS), RVOS replaces mask prompts with language, enabling more natural user interaction but introducing additional ambiguity under occlusion, viewpoint change, and multi-instance clutter. Consequently, RVOS progress is commonly measured on benchmarks such as Ref-DAVIS17 [5], MeViS [7], and ReVOS [8].

Recent advances have combined foundation segmentation with multimodal large language models (MLLMs), enabling densely grounded understanding in videos. In particular, Sa2VA couples an MLLM (LLaVA-style) with SAM2 to produce grounded [SEG] tokens and high-quality masks [1]–[3], [10]. However, Sa2VA’s inference relies on a small fixed set of keyframes to establish grounding, which uses the first five frames of the input video as keyframes for prompting and then propagates masks through the remaining frames [1]. In long or dynamic videos, the initial grounding can become stale when the object composition changes (e.g., target disappears/re-appears, new distractors enter, or layout shifts), often manifesting as false positives or drift.

Motivated by this limitation, we study the following core problem: how to refresh grounded segmentation only when the video content meaningfully changes, without incurring the cost of dense re-prompting. We propose an Event-Driven Refresh + Recurrence Memory (EDRRM) enhancement to Sa2VA [1] (Figs. 1 and 2). Our method monitors the video stream with an object tracker and simple change cues (e.g., instance births/deaths and coarse layout shifts) to detect stable change points and trigger a re-invocation of Sa2VA on a compact frame batch around the event. In addition, we introduce a recurrence memory that stores embeddings of recently disappeared instances and matches them upon re-appearance using CLIP similarity, enabling anchor-frame retrieval to re-condition Sa2VA and reduce identity switches [4], [21].

We evaluate on RVOS benchmarks including Ref-DAVIS17 [5], [17], MeViS [7], and ReVOS [8], and compare against two practical sampling baselines: (i) fixed-window reprompting and (ii) FrameDiff-SSIM change detection based on structural similarity [9]. Overall, our approach keeps Sa2VA’s strong grounded segmentation backbone while reducing stalegrounding failures by re-invoking Sa2VA only at event-driven change points and recurrence-driven re-appearances. Crucially, EDRRM frames re-grounding as a scheduling and control problem: rather than applying a fixed inference, it explicitly decides when to invoke grounding based on object-composition change signals, cleanly separating the scheduling mechanism from the segmentation backbone itself.

## II. RELATED WORK

RVOS models and benchmarks. RVOS requires segmenting the instance referred by a natural-language expression across time, which demands both correct grounding and robust temporal association under occlusion, distractors, and motioncentric cues. Representative CNN/transformer-based RVOS methods perform multimodal interaction between language and video features to improve temporal reasoning and instance discrimination [6], [11]–[13], [15], [16]. Efficiency-oriented settings have also been explored via online/semi-online inference to support streaming with competitive accuracy [14]. Progress is driven by diverse benchmarks covering static and motionheavy expressions, including Ref-DAVIS17 [5], MeViS [7], and ReVOS [8], which emphasize motion expressions and highlights temporal grounding failures.

Foundation models and grounded segmentation with MLLMs. Promptable foundation segmentation (SAM) and its video extension (SAM2) enable strong mask prediction with minimal supervision [2], [3], while MLLMs such as LLaVA improve instruction-following visual understanding [10]. Sa2VA integrates SAM2 with an LLaVA-style MLLM to produce grounded [SEG] tokens for dense understanding in images and videos [1]. However, Sa2VA-style pipelines typically ground the target from a compact set of keyframes and then rely on propagation, which can become brittle when video content evolves (e.g., target disappearance/re-appearance, distractor entry, or layout shifts), motivating selective re-grounding rather than purely relying on long-range propagation.

Temporal memory, association, and robustness. Long-term VOS literature shows that explicit memory and association are critical for stable propagation in long videos [18]–[20]. In parallel, retrieval-style re-identification using shared embedding spaces (e.g., CLIP) supports matching instances across time [4], and tracking/association modules such as MASA provide strong temporal continuity cues [21]. Robust RVOS further studies semantic mismatch cases where the query target is absent or ambiguous, motivating gating/consensus beyond perframe confidence [22]. Our work draws from these directions: instead of refreshing at fixed intervals or low-level framedifference heuristics, we propose event-driven re-grounding with recurrence-aware anchors to reduce stale-grounding drift in Sa2VA-style inference.

Adaptive inference and temporal decision policies. Beyond fixed-schedule inference, a growing line of work frames temporal re-inference as a scheduling or control problem. Adaptive frame sampling methods for video recognition selectively process only informative frames based on content driven cues [23], [24], reducing redundant computation while preserving accuracy. Confidence-based early-exit frameworks defer computation to only those inputs that require deeper processing [24]. These methods share the core insight of EDRRM: rather than applying a uniform inference schedule, the computation is conditioned on the input signal. Our work extends this idea to RVOS by treating re-grounding as an event-triggered scheduling decision, where object-composition changes derived from the tracking stream determine when Sa2VA is re-invoked, rather than relying on fixed windows or low-level appearance differences.

## III. METHODOLOGY

Fig. 1 illustrates the end-to-end process flow of our eventdriven referring video object segmentation (RVOS) system, and

Fig. 2 details the proposed recurrence memory mechanism. Our goal is to reduce redundant segmentation calls and improve temporal robustness by triggering Sa2VA [1] segmentation only when object-level changes (events) are detected, while additionally handling re-appearance (recurrence) via CLIPbased matching.

## A. Problem Setup and Notation

Let a video be a sequence of RGB frames $\scriptstyle \left\{ \mathbf { I } _ { t } \right\} _ { t = 0 } ^ { T - 1 }$ and a user referring query be q (e.g., “segment the owl and the person”). Our objective is to produce a binary mask sequence $\{ \bar { \mathbf { M } } _ { t } \} _ { t = 0 } ^ { T - 1 }$ where $\mathbf { \bar { M } } _ { t } \in \{ 0 , 1 \} ^ { \mathbf { \bar { H } } \times W }$ indicates the target object(s) in frame t.

We use an object tracker to obtain a per-frame track state:

$$
{ \cal S } _ { t } = \{ ( i d _ { i } ^ { t } , ~ \ell _ { i } ^ { t } , ~ { \bf b } _ { i } ^ { t } ) \} _ { i = 1 } ^ { N _ { t } } ,\tag{1}
$$

where $i d _ { i } ^ { t }$ is the track identity, $\ell _ { i } ^ { t }$ is the (open-vocabulary) class label, and $\mathbf { b } _ { i } ^ { t } = ( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ is an axis-aligned bounding box.

## B. System Overview

As shown in Fig. 1, the pipeline consists of: (i) query decoding to produce cleaned tokens for tracking and segmentation, (ii) tracking-driven event scoring to decide segmentation boundaries, (iii) recurrence memory to detect re-appearances and retrieve anchor frames (Fig. 2), (iv) boundary manager to assemble a frame batch for Sa2VA, (v) identifiability gate to prevent hallucinated segmentation, and (vi) mask prediction with Sa2VA on selected frames only. Finally, masks are realigned back to the original timeline.

Implementation note. In our implementation, the querydecoding LLM (InternLM2.5-7B) and the identifiability MLLM (InternVL2.5-8B) are both instruction-following components already packaged within the Sa2VA-8B release. Thus, EDRRM does not require the deployment of additional large models beyond Sa2VA-8B; it only changes when Sa2VA is invoked and how input frame batches are constructed.

## C. Query Decoding for Tracking and Segmentation

Given q, an LLM (InternLM2.5-7B in our implementation) generates a cleaned token set V (“vocab\_texts”) and a segmentation prompt p used by the RVOS model (Sa2VA). The tokens V are passed to the tracker to improve detection/association for target-relevant objects, while p is used to drive segmentation. This stage corresponds to the “Target object decoding” block in Fig. 1.

## D. Recurrence Memory with CLIP Embeddings

Recurrence memory aims to detect when a newly appearing track at time t is a re-appearance of a previously disappeared object. Fig. 2 illustrates the memory design.

1) Active/Dead Memory Buffers: We maintain:

• Active memory A: a dictionary mapping active track IDs to their CLIP embeddings and metadata.

• Dead memory D: a FIFO deque of recently disappeared tracks, pruned by a time-to-live (TTL).

Each memory entry stores $( \ell , \mathbf { e } , t _ { \mathrm { l a s t } } , t _ { \mathrm { a n c } } ) ;$ : label, normalized embedding, last-seen timestamp, and an anchor frame index used later for re-conditioning.

![](images/4274fdc2d4362bc24ad767c8ec7df944ca2175d10f17b4848c285d42b6285c04.jpg)  
Fig. 1. Process flow of the proposed pipeline.

2) CLIP Crop Embedding: For a cropped object image x from frame $\mathbf { I } _ { t } .$ , we compute:

$$
\mathbf { e } ( \mathbf { x } ) = \frac { f _ { \mathrm { C L I P } } ( \mathbf { x } ) } { \| f _ { \mathrm { C L I P } } ( \mathbf { x } ) \| _ { 2 } + \epsilon } ,\tag{2}
$$

where $f _ { \mathrm { C L I P } } ( \cdot )$ is the CLIP image encoder. Embeddings are refreshed periodically (every $\Delta _ { \mathrm { u p d } }$ frames) to balance cost and robustness.

3) Birth/Death Sets: Let ${ \mathcal { T } } _ { t } = \{ i d _ { i } ^ { t } \}$ be the set of track IDs at time t. We define:

$$
\boldsymbol { B } _ { t } = \boldsymbol { \mathcal { I } } _ { t } \setminus \boldsymbol { \mathcal { T } } _ { t - 1 } , \quad \boldsymbol { \mathcal { D } } _ { t } = \boldsymbol { \mathcal { T } } _ { t - 1 } \setminus \boldsymbol { \mathcal { T } } _ { t } ,\tag{3}
$$

as the birth and death ID sets, respectively. When id $\in \mathcal { D } _ { t }$ , we move its embedding entry from A to the dead deque D.

4) Gated Cosine Similarity Matching: For each birth id $\in \mathcal { B } _ { t }$ we embed its crop $\mathbf { x } _ { i d } ^ { t }$ and match it against dead candidates of the same label:

$$
s ( i d , j ) = \big \langle \mathbf { e } ( \mathbf { x } _ { i d } ^ { t } ) , \ \mathbf { e } _ { j } \big \rangle , \quad \mathrm { o n l y ~ i f ~ } \ell _ { i d } ^ { t } = \ell _ { j } ,\tag{4}
$$

where $\mathbf { e } _ { j }$ is the stored embedding of dead entry j. We select $j ^ { \star } = \arg$ max s(id, j) and declare recurrence if

$$
s ( i d , j ^ { \star } ) \geq \sigma ,\tag{5}
$$

where $\sigma$ is a similarity threshold. Each successful match increments a recurrence count $R _ { t }$ and retrieves an anchor frame $t _ { \mathrm { a n c } } ^ { ( j ^ { \star } ) }$

$$
R _ { t } = \sum _ { i d \in \mathcal { B } _ { t } } \mathbb { I } \big [ s ( i d , j ^ { \star } ) \ge \sigma \big ] , \quad \mathcal { A } _ { t } ^ { \mathrm { a n c } } = \{ t _ { \mathrm { a n c } } ^ { ( j ^ { \star } ) } \} .\tag{6}
$$

These anchors are used by the boundary manager to augment the Sa2VA batch (Fig. 2, “Segment Builder”).

5) TTL Pruning: Dead memory D is intended to represent recently disappeared tracks for reliable re-appearance matching; keeping old entries increases false matches (appearance drift) and expands the search set during cosine matching. Thus, we enforce a time-to-live (TTL) window: for a dead entry j with last-seen time $t _ { \mathrm { l a s t } } ^ { ( j ) }$ , we remove it when

$$
\begin{array} { r } { ( t - t _ { \mathrm { l a s t } } ^ { ( j ) } ) > \mathrm { T T L } . } \end{array}\tag{7}
$$

Since $\mathcal { D }$ is stored as a time-ordered deque, pruning is efficient (pop-from-front) and keeps recurrence matching bounded and temporally local.

Failure modes and mitigation. CLIP-based cosine matching can produce false positives under heavy occlusion, significant viewpoint change, or when same-class objects serve as distractors (e.g., multiple persons in a crowded scene), causing the similarity score to be inflated for an incorrect match and potentially triggering a spurious recurrence event. Two mechanisms mitigate this risk. First, label gating restricts dead-entry candidates to those sharing the same class label as the new birth, substantially reducing the candidate pool. Second, and most critically, the identifiability gate (Sec. III-G) serves as a second stage safeguard: if the target is not unambiguously identifiable in the upcoming segment frames, the gate suppresses the refresh call regardless of the recurrence signal, preventing hallucinated masks even when CLIP matching produces a false positive.

## E. Tracking-Driven Event Score

We compute an event score $E _ { t }$ from tracking state changes and recurrence signals, then apply an EMA smoother to produce a stable trigger signal.

![](images/8a27c3dc526f8d675d733286325c6118f825a3c2bf685b69f66c8ba34604f58b.jpg)  
Fig. 2. Recurrence memory flowchart.

1) Birth/Death Counts: We first compute:

$$
b _ { t } = \vert B _ { t } \vert , \quad d _ { t } = \vert { \mathcal { D } } _ { t } \vert .\tag{8}
$$

2) Class Composition Change: We build a normalized class histogram $\mathbf { h } _ { t } \in \mathbb { R } ^ { C }$ :

$$
h _ { t } ( c ) = \frac { 1 } { N _ { t } } \sum _ { i = 1 } ^ { N _ { t } } \mathbb { I } [ \ell _ { i } ^ { t } = c ] , \quad c \in \{ 1 , \ldots , C \} ,\tag{9}
$$

and define the histogram L1 difference:

$$
\Delta H _ { t } = \| \mathbf h _ { t } - \mathbf h _ { t - 1 } \| _ { 1 } .\tag{10}
$$

3) Spatial Layout Change via Grid Occupancy: We discretize the image into a $G \times G$ grid and accumulate a normalized occupancy vector $\mathbf { g } _ { t } \in \bar { \mathbb { R } } ^ { G ^ { 2 } }$ by assigning each box center $\left( c _ { x } , c _ { y } \right)$ to a grid cell. The layout change is:

$$
\begin{array} { r } { \Delta L _ { t } = \| \mathbf { g } _ { t } - \mathbf { g } _ { t - 1 } \| _ { 1 } . } \end{array}\tag{11}
$$

4) Unified Event Score: Let $R _ { t }$ be the recurrence count from Sec. III-D. We define:

$$
E _ { t } = w _ { b } b _ { t } + w _ { d } d _ { t } + w _ { c } \Delta H _ { t } + w _ { l } \Delta L _ { t } + w _ { r } R _ { t } ,\tag{12}
$$

where $\{ w _ { b } , w _ { d } , w _ { c } , w _ { l } , w _ { r } \}$ are scalar weights. In our implementation, recurrence is deliberately weighted higher $( w _ { r } )$ because a confident re-appearance often warrants immediate refresh.

5) EMA Smoothing and Cooldown Trigger: We smooth $E _ { t }$ using an exponential moving average (EMA):

$$
\tilde { E } _ { t } = \alpha E _ { t } + ( 1 - \alpha ) \tilde { E } _ { t - 1 } ,\tag{13}
$$

where $\alpha \in ( 0 , 1 ]$ controls responsiveness. A refresh trigger fires if:

$$
\tilde { E } _ { t } \geq \tau \ \wedge \ \subset \mathrm { o o l d o w n } = 0 ,\tag{14}
$$

where $\tau$ is the event threshold. After triggering, we set cooldown← K to suppress immediate re-triggers for K frames. Additionally, we enforce a minimum segment length min\_chunk (Sec. III-F) to avoid excessively short segments. Recurrence triggers force refresh regardless of $\tilde { E } _ { t }$ (and optionally regardless of cooldown), since re-appearance is a high-risk drift event.

## F. Boundary Manager and Segment Construction

The boundary manager converts triggers into segmentation segments. Let accepted boundaries be

$$
0 = b _ { 0 } < b _ { 1 } < \dots < b _ { K } = T ,\tag{15}
$$

where $b _ { k }$ is accepted only if it satisfies the minimum spacing constraint:

$$
b _ { k } - b _ { k - 1 } \geq L _ { \operatorname* { m i n } } ,\tag{16}
$$

with $L _ { \mathrm { m i n } }$ denoting min\_chunk. For each segment $[ b _ { k } , b _ { k + 1 } )$ we build a Sa2VA input frame index set:

$$
\mathcal { I } _ { k } = \Big ( \{ b _ { k } - p , \dotsc , b _ { k + 1 } + q \} \cap [ 0 , T - 1 ] \Big ) \ \cup \ \mathcal { A } _ { b _ { k + 1 } } ^ { \mathrm { a n c } } ,\tag{17}
$$

Table 1. Frozen default event-score weights (kept fixed across all benchmarks).
<table><tr><td>Parameter</td><td>Default</td><td>Role</td></tr><tr><td>wb (wb)</td><td>1.0</td><td>Weight for instance births term in the event score.</td></tr><tr><td>wd (wd)</td><td>1.0</td><td>Weight for instance deaths term in the event score.</td></tr><tr><td> $w _ { c }$  (wc)</td><td>0.5</td><td>Weight for class-histogram change term in the event score.</td></tr><tr><td>wl (w1)</td><td>0.5</td><td>Weight for spatial layout-grid change term in the event score.</td></tr><tr><td>wr (wr)</td><td>2.0</td><td>Weight for recurrence-count term (strong trigger).</td></tr></table>

Algorithm 1 Event-Driven Sa2VA Inference with Recurrence Algorithm 2 Recurrence Memory Update and Anchor Retrieval   
Anchors Require: Time t, frame $\mathbf { I } _ { t } ,$ previous state $S _ { t - 1 }$ , current state   
Require: Video frames $\begin{array} { r } { \{ \mathbf { I } _ { t } \} _ { t = 0 } ^ { T - 1 } } \end{array}$ , query $q ,$ thresholds $\tau , \sigma ,$ $S _ { t } .$ , TTL, similarity threshold σ   
EMA $\alpha ,$ cooldown $K ,$ , min length $L _ { \mathrm { m i n } }$ Ensure: Recurrence count $R _ { t } ,$ anchor set $\mathcal { A } _ { t } ^ { \mathrm { a n c } }$   
Ensure: Predicted masks $\begin{array} { r } { \{ \hat { \mathbf { M } } _ { t } \} _ { t = 0 } ^ { T - \top } } \end{array}$ 1: Prune dead deque by TTL: remove if $( t - t _ { \mathrm { l a s t } } ) > \mathrm { T T L }$   
1: Decode $( p , \mathcal { V } ) \gets \mathrm { L L M } ( q )$ 2: Compute births $B _ { t }$ and deaths $\mathcal { D } _ { t }$ from track ID sets   
2: Initialize boundary $b _ { 0 } \gets 0 , s \gets 0$ (segment start), $\tilde { E } _ { - 1 } \gets$ 3: Periodically refresh embeddings for active tracks (every   
0 $\Delta _ { \mathrm { u p d } }$ frames)   
3: for $t = 0$ to $T - 1$ do 4: for each $i d \in \mathcal { D } _ { t }$ do   
4: S<sub>t</sub> ← Tracker $\mathbf { \boldsymbol { I } } _ { t } ; \mathcal { V } )$ 5: Move $( \ell , \mathbf { e } , t _ { \mathrm { l a s t } } , t _ { \mathrm { a n c } } )$ from active dict to dead deque   
5: $( R _ { t } , \mathcal { A } _ { t } ^ { \mathrm { a n c } } ) \gets$ RecurrenceUpdate $( t , \underline { { \mathbf { I } } } _ { t } , \boldsymbol { S } _ { t - 1 } , \boldsymbol { S } _ { t } )$ 6: end for   
6: Compute $E _ { t }$ via Eq. (12); update $\tilde { E } _ { t }$ via Eq. (13) 7: $R _ { t } \gets 0 , \mathcal { A } _ { t } ^ { \mathrm { a n c } } \gets \emptyset$   
7: $\mathrm { t r i g }  [ \tilde { E } _ { t } \geq \tau \land \mathsf { c o o l d o w n } = 0 ] \lor [ R _ { t } > 0 ]$ 8: for each $i d \in B _ { t }$ do   
8: if trig $\Lambda \left( t - s \right) \geq L _ { \operatorname* { m i n } }$ then 9: Embed crop $\mathbf { e } _ { i d } \gets \mathrm { C L I P } ( \mathrm { c r o p } ( \mathbf { I } _ { t } , \mathbf { b } _ { i d } ^ { t } ) )$   
9: Build batch indices $\mathcal { I }$ 10: Find best dead match $j ^ { \star }$   
10: $g \gets \mathrm { I d e n t i f i a b l e } ( \{ \mathbf { I } _ { u } \} _ { u \in [ s , t ) } )$ $\scriptstyle { \arg \operatorname* { m a x } _ { j \in \mathrm { D e a d } : \ell _ { j } = \ell _ { i d } ^ { t } } \left. \mathbf { e } _ { i d } , \mathbf { e } _ { j } \right. }$   
11: if $g = \Upsilon \in S$ then 11: if $\langle \mathbf { e } _ { i d } , \mathbf { \check { e } } _ { j ^ { \star } } \rangle \geq \check { \sigma }$ then   
12: Run Sa2VA on $\{ \mathbf { I } _ { u } \} _ { u \in \mathcal { I } }$ with prompt p to get 12: $R _ { t } \gets \operatorname { \bar { \cal R } } _ { t } + 1 ; \operatorname { \mathcal { A } } _ { t } ^ { \mathrm { a n c } } \gets \operatorname { \mathcal { A } } _ { t } ^ { \mathrm { a n c } } \cup \{ t _ { \mathrm { a n c } } ^ { ( j ^ { \star } ) } \}$   
$\{ \hat { \mathbf { M } } _ { u } \} _ { u \in \mathcal { I } }$ 13: end if   
13: else 14: Register birth as active with its embedding and anchor t   
14: Set $\hat { { \bf M } } _ { u } \gets { \bf 0 }$ for $u \in \mathcal I$ 15: end for   
15: end if 16: return $R _ { t } , A _ { t } ^ { \mathrm { a n c } }$   
16: Re-align and write output masks for core frames   
$u \in [ s , t )$   
17: Update boundary: $s \gets t ;$ set cooldown← K object is present and unambiguously identifiable. The gate object is present and unambiguously identifiable. The gate   
18: end if   
returns returns $g _ { k } { = } \mathrm { y } { \in } S$ only upon a positive confirmation; otherwise only upon a positive confirmation; otherwise   
19: end for   
$g _ { k } { = } \mathtt { n o }$ and the segment receives empty masks without invoking and the segment receives empty masks without invoking   
20: Process final segment $[ s , T )$ similarly Sa2VA. This design is intentionally conservative: a missed Sa2VA. This design is intentionally conservative: a missed

where $p , q$ are pre/post context lengths and $\mathcal { A } _ { b _ { k + 1 } } ^ { \mathrm { a n c } }$ are anchor 1 frames (possibly empty) obtained from recurrence matching (Fig. 1, "Boundary Manager").

$$
g _ { k } \in \{ \mathrm { y e s } , \mathrm { n o } \} .
$$

Before running Sa2VA [1], we perform an identifiability check (InternVL2.5-8B in our implementation). Given a small subset of frames sampled from the core segment $[ b _ { k } , b _ { k + 1 } )$ the gate outputs a binary decision:

(18)

If $g _ { k } = \mathtt { n o }$ , we return an empty mask sequence for the segment, preventing false positives when the target is absent or not identifiable (Fig. 1, “Identifiability $\mathrm { G a t e } ^ { \prime \prime } )$

Concretely, the gate is implemented as a visual questionanswering (VQA) prompt to InternVL2.5-8B: given $n _ { \mathrm { g a t e } } { = } 3$ uniformly sampled frames from $[ b _ { k } , b _ { k + 1 } )$ and the original referring query q, the model is asked whether the described

## G. Identifiability Gate to Prevent Hallucinated Masks

segment (false negative) is preferable to a hallucinated mask (false positive) when the target is absent or ambiguous. The gate requires no additional threshold tuning beyond the VQA model’s own output distribution. The practical impact is quantified in Sec. IV-F, where per-dataset rejection rates confirm that the gate actively suppresses 20–26% of candidate refresh calls across benchmarks.

## H. Sa2VA Mask Prediction and Output Alignment

If the segment passes the gate, we invoke Sa2VA on the batch frames $\{ \mathbf { I } _ { t } \} _ { t \in \mathcal { T } _ { k } }$ with prompt p:

$$
\{ \hat { \mathbf { M } } _ { t } \} _ { t \in \mathcal { T } _ { k } } = f _ { \mathrm { S a 2 V A } } \left( \{ \mathbf { I } _ { t } \} _ { t \in \mathcal { T } _ { k } } , \ p \right) .\tag{19}
$$

Since $\mathcal { T } _ { k }$ may include anchors and context frames, we re-align the predicted masks to the contiguous core indices $t \in [ b _ { k } , b _ { k + 1 } )$ by indexing into the batch result (implementation detail in our runner). The global output is the concatenation over all segments:

$$
\hat { \mathbf { M } } _ { t } = \hat { \mathbf { M } } _ { t } ^ { ( k ) } , \quad \forall t \in [ b _ { k } , b _ { k + 1 } ) .\tag{20}
$$

Table 2. EDRRM Configuration Ablation
<table><tr><td>Config</td><td>thr_track</td><td>ema_alpha</td><td>event_thr</td><td>cooldown</td><td>min_chunk</td><td>sim_thr</td><td>ttl</td><td>recurrence_cooldown</td><td>pre_ctx</td><td>post_ctx</td><td>DAVIS [5]</td><td>MEVIS [7]</td><td>REVOS [8]</td></tr><tr><td>edrrm-cfg-1</td><td>0.5</td><td>0.3</td><td>2.5</td><td>14</td><td>14</td><td>0.34</td><td>110</td><td>25</td><td>2</td><td>2</td><td>76.78</td><td>61.50</td><td>65.34</td></tr><tr><td>edrrm-cfg-2</td><td>0.5</td><td>0.45</td><td>2.0</td><td>10</td><td>10</td><td>0.34</td><td>90</td><td>25</td><td></td><td></td><td>76.78</td><td>61.89</td><td>65.34</td></tr><tr><td>edrrm-cfg-3</td><td>0.5</td><td>0.25</td><td>2.8</td><td>16</td><td>16</td><td>0.36</td><td>240</td><td>50</td><td></td><td></td><td>76.77</td><td>61.43</td><td>65.34</td></tr><tr><td>edrrm-cfg-4</td><td>0.42</td><td>0.45</td><td>2.0</td><td>10</td><td>10</td><td>0.42</td><td>110</td><td>35</td><td>222</td><td>222</td><td>76.76</td><td>61.08</td><td>65.33</td></tr><tr><td>edrrm-cfg-5</td><td>0.25</td><td>0.65</td><td>1.4</td><td>4</td><td>4</td><td>0.22</td><td>240</td><td>8</td><td>2</td><td>2</td><td>76.58</td><td>63.13</td><td>65.75</td></tr><tr><td>edrrm-cfg-6</td><td>0.3</td><td>0.7</td><td>1.6</td><td>6</td><td>6</td><td>0.20</td><td>300</td><td>10</td><td>4</td><td></td><td>76.21</td><td>62.43</td><td>65.60</td></tr><tr><td>edrrm-cfg-7</td><td>0.3</td><td>0.55</td><td>1.5</td><td>4</td><td>6</td><td>0.18</td><td>180</td><td>6</td><td>2</td><td>42</td><td>76.52</td><td>62.62</td><td>65.53</td></tr><tr><td>edrrm-cfg-8</td><td>0.25</td><td>0.45</td><td>1.7</td><td>8</td><td>8</td><td>0.22</td><td>400</td><td>12</td><td>4</td><td>4</td><td>76.60</td><td>61.92</td><td>66.16</td></tr><tr><td>edrrm-cfg-9</td><td>0.3</td><td>0.8</td><td>1.2</td><td>2</td><td>4</td><td>0.20</td><td>240</td><td>4</td><td>8</td><td>8</td><td>76.84</td><td>60.92</td><td>65.55</td></tr></table>

## I. Algorithms

Algorithm 1 summarizes the full event-driven inference loop (Fig. 1): for each frame, we update tracking, compute the event score and EMA, and accept a boundary only when the trigger fires and the minimum segment length constraint holds; the boundary manager then builds the Sa2VA batch indices (core segment plus optional context and recurrence anchors) and runs an identifiability gate before invoking Sa2VA, finally re-aligning batch masks to the core timeline. Algorithm 2 details recurrence memory (Fig. 2): deaths move from active to dead, dead entries are TTL-pruned (Eq. 7), and each birth is embedded with CLIP and matched (label-gated cosine) against dead candidates; successful matches yield a recurrence count and anchor frames that are injected into the next segment batch.

## J. Parameters and Defaults

Table 1 reports the parameter defaults that we keep frozen across all dataset benchmarks: the event-score weighting parameters $( w _ { b } , w _ { d } , w _ { c } , w _ { l } , w _ { r } )$ . These weights define the relative contribution of instance births, instance deaths, classcomposition change, layout change, and recurrence signals in our event score, ensuring a consistent definition of “eventfulness” across Ref-DAVIS17 [5], MeViS [7], and ReVOS [8].

## K. Computational Cost

At time $t ,$ event score computation is $O ( N _ { t } + G ^ { 2 } + C )$ for births/deaths, grid occupancy, and histogram updates. Recurrence matching cost is dominated by comparing each birth embedding to dead candidates; with label gating, the worst-case cost is $O ( | B _ { t } | \cdot | D | )$ but typically much smaller due to TTL pruning and label filtering. The overall system reduces expensive Sa2VA calls by segmenting only selected batches, yielding an efficiency-accuracy trade-off governed by $\tau , \alpha , K , L _ { \operatorname* { m i n } } , \sigma .$ , and TTL.

## IV. EXPERIMENTS

We evaluate Event-Driven Refresh + Recurrence Memory (EDRRM) on Ref-DAVIS17 [5], MeViS [7], and ReVOS [8] using J&F as the primary metric, and compare against two refresh schedulers: Fixed Window and FrameDiff-SSIM [9]. All experiments are run on a machine with 4× NVIDIA RTX 8000 GPUs. We first visualize how event refresh selects boundaries and how each score component contributes (Sec. IV-B), then report configuration ablations (Sec. IV-C). Next, we analyze the accuracy-cost trade-off via the average number of Sa2VA calls [1] and Pareto optimality (Sec. IV-H), followed by refreshrate budget evaluation (Sec. IV-I). Finally, we present main results under global-tuned vs. per-dataset oracle tuning and provide a qualitative comparison.

Table 3. FrameDiff-SSIM Configuration Ablation
<table><tr><td>Config</td><td>threshold</td><td>cooldown</td><td>min_length</td><td>DAVIS [5]</td><td>MEVIS [7]</td><td>REVOS [8]</td></tr><tr><td>ssim-cfg-1</td><td>0.95</td><td>4</td><td>4</td><td>76.83</td><td>62.48</td><td>65.25</td></tr><tr><td>ssim-cfg-2</td><td>0.93</td><td>8</td><td>8</td><td>77.39</td><td>62.24</td><td>65.55</td></tr><tr><td>ssim-cfg-3</td><td>0.90</td><td>8</td><td>12</td><td>76.85</td><td>61.76</td><td>65.41</td></tr><tr><td>ssim-cfg-4</td><td>0.87</td><td>12</td><td>16</td><td>76.79</td><td>61.59</td><td>65.52</td></tr><tr><td>ssim-cfg-5</td><td>0.82</td><td>20</td><td>32</td><td>76.47</td><td>59.85</td><td>65.43</td></tr><tr><td>ssim-cfg-6</td><td>0.80</td><td>8</td><td>8</td><td>77.24</td><td>62.35</td><td>65.21</td></tr><tr><td>ssim-cfg-7</td><td>0.30</td><td>8</td><td>8</td><td>76.48</td><td>59.91</td><td>65.22</td></tr></table>

Table 4. Fixed Window Configuration Ablation
<table><tr><td>Config</td><td>length</td><td>DAVIS [5]</td><td>MEVIS [7]</td><td>REVOS [8]</td></tr><tr><td>window-cfg-1</td><td>8</td><td>77.45</td><td>62.34</td><td>65.55</td></tr><tr><td>window-cfg-2</td><td>16</td><td>76.53</td><td>55.25</td><td>65.88</td></tr><tr><td>window-cfg-3</td><td>32</td><td>76.99</td><td>59.85</td><td>65.40</td></tr><tr><td>window-cfg-4</td><td>64</td><td>75.94</td><td>58.92</td><td>65.02</td></tr><tr><td>window-cfg-5</td><td>128</td><td>76.31</td><td>59.00</td><td>65.40</td></tr></table>

## A. Implementation Details

All experiments are run on 4× NVIDIA RTX 8000 GPUs. The object tracker is MASA [21] with an open-vocabulary detector backbone; thr\_track controls the minimum track confidence and is swept from 0.25 to 0.50 in our sensitivity ablation (Sec. IV-E). CLIP embeddings for recurrence matching use the ViT-B/32 encoder [4] with object crops resized to 224×224. The identifiability gate uses InternVL2.5-8B with $n _ { \mathrm { g a t e } } { = } 3$ sampled frames per segment. Query decoding uses InternLM2.5-7B (packaged within Sa2VA-8B), so EDRRM requires no additional large models beyond Sa2VA-8B. The default global-tuned configuration (edrrm-cfg-5) uses: thr\_track=0.25, α=0.65, τ=1.4, cooldown=4, min\_chunk=4, sim\_thr=0.22, TTL=240, recurrence\_cooldown=8, pre\_ctx=2, and post\_ctx=2. All hyperparameters were tuned by grid search over the MeViS validation split and then frozen for DAVIS and ReVOS under the global-tuned protocol, or independently selected per dataset under the oracle-tuned protocol. No dataset-specific feature engineering beyond threshold tuning was applied.

## B. Event Refresh Timeline and Score Decomposition

Fig. 3 illustrates how our event-driven scheduler produces refresh boundaries over time and why each boundary is selected. In Fig. 3(a), frames surrounding a boundary show a clear and persistent change in the scene/object composition, which motivates re-grounding rather than continuing long-range propagation. Fig. 3(b) plots the raw event score $E ( t )$ together with its EMA-smoothed signal; a boundary is accepted when the EMA exceeds the threshold and the minimum segmentlength constraint is satisfied, while recurrence triggers are marked when a previously disappeared instance is matched and yields anchor frames. Finally, Fig. 3(c) decomposes E(t) into its weighted components (births, deaths, class/layout change, and recurrence), showing that boundary locations align with large contributions from one or more cues, thus providing interpretability for the refresh decisions.

- Boundary frames were detected due to object composition change  
(a) Boundary examples  
![](images/d5ef217cc1d4c4772378f7b048c5c55e784995f1050b21a1984dcb5bea5165a5.jpg)  
- Before & After refresh boundary

![](images/f908658718ec917d3d5afb640750c0cb3d41275141dc85f03cf38527e4360725.jpg)

(b) Event score timeline with refresh boundaries  
![](images/a72e0de6a7bf40352dc873935eafed8b05aaca97508ef84b98b32791c45b954d.jpg)

(c) Event score decomposition  
![](images/bef9c0912abb05be43ff9c2a79074769e781f35702fd63796dad419d3db86646.jpg)  
Fig. 3. Event-refresh visualization and interpretability.

## C. Configuration Ablation and Baseline Comparison

Tables 2, 3, and 4 summarize our configuration sweeps for the proposed Event-Driven Refresh + Recurrence Memory (EDRRM) and two competing scheduling frameworks: FrameDiff-SSIM [9] and Fixed Window. For

EDRRM (Table 2), we jointly vary the tracker filtering and event scheduler (thr\_track, α, τ , cooldown, min\_chunk) together with recurrence controls (sim\_thr, ttl, recurrence\_cooldown) and optional context (pre\_ctx/post\_ctx); the best per-dataset performance is achieved by edrrm-cfg-9 on Ref-DAVIS17 (76.84), edrrm-cfg-5 on MeViS (63.13), and edrrm-cfg-8 on ReVOS (66.16). For FrameDiff-SSIM (Table 3), we sweep the SSIM threshold and temporal constraints, obtaining best Ref-DAVIS17 and ReVOS with ssim-cfg-2 (77.39, 65.55) and best MeViS with ssim-cfg-1 (62.48). For Fixed Window (Table 4), the window length controls refresh frequency, where window-cfg-1 gives the best Ref-DAVIS17/MeViS (77.45, 62.34) and window-cfg-2 gives the best ReVOS (65.88). Overall, EDRRM achieves the strongest results on MeViS and ReVOS among the evaluated frameworks, while remaining competitive on Ref-DAVIS17, highlighting the benefit of refreshing at content-driven change points.

## D. Component Ablation

To isolate the contribution of each module, Table 5 reports J&F and average #refresh (#ref) for eight variants on all three benchmarks. The Pareto frontier in Fig. 4 visualizes accuracy versus refresh cost per variant, and Fig. 5 shows J&F as a function of integer refresh budget across the three methods.

Several patterns emerge from Table 5. First, comparing Event Score only against Recurrence only directly separates the two core contributions: Event Score only achieves 76.23/61.78/66.22 while Recurrence only achieves 76.68/62.42/66.15, and the full EDRRM combining both reaches 76.84/63.13/66.16, confirming that gains arise from both general event-driven scheduling and specific re-appearance handling, not from either alone. Second, removing recurrence memory reduces MeViS J&F from 63.13 to 61.79 while also reducing refresh cost (7.89→5.91 #ref), indicating that recurrence events trigger useful additional refreshes that recover from re-appearance failures. Third, the identifiability gate’s contribution is primarily in false-positive suppression (discussed further in Sec. IV-F) rather than bulk J&F, as evidenced by the modest raw J&F difference when removing it.

## E. Tracker Sensitivity Analysis

Since all event signals derive from tracker output, we analyze how tracking strictness affects EDRRM. Table 6 sweeps thr\_track from 0.25 (Very noisy: many tracks accepted) to 0.50 (Very Strict: only high-confidence tracks), effectively modeling tracker reliability. Fig. 6 shows J&F and average #refresh jointly as thr\_track increases.

At low thr\_track (Very noisy), more tracks generate more events and higher refresh counts, recovering more re-appearance events and yielding higher J&F on MeViS (62.53) and ReVOS (65.96). At high thr\_track (Very Strict), fewer tracks survive, reducing refresh calls dramatically (to 1.25 on DAVIS) but suppressing valid events. Critically, J&F degrades gracefully: the accuracy range across all thr\_track values is at most 1.3 J&F points on any dataset, demonstrating that EDRRM is not brittle to this parameter. This proxy analysis models the effect of tracker reliability (e.g., missed detections or ID switches reducing track completeness): even under degraded tracking, accuracy remains within a tight and predictable band. While direct injection of ID-switch or missed-detection failures is left for future work, the thr\_track sweep captures the accuracyrobustness trade-off under degraded track completeness as a tractable proxy.

## F. Identifiability Gate Analysis

Table 7 reports the total candidate refresh segments, the number suppressed by the gate $( g _ { k } = \mathtt { n o } )$ , and the gate rejection rate per benchmark.

The gate suppresses 20–26% of candidate refresh calls per dataset, confirming that a substantial fraction of triggered events correspond to segments where the target is absent or unidentifiable. Without the gate, these segments would invoke Sa2VA unnecessarily, generating hallucinated masks. The higher rejection rate on ReVOS (25.60%) versus DAVIS (20.56%) is consistent with ReVOS being a more dynamic benchmark with more frequent target absence. A gate false reject suppresses a segment where the target is actually present and reduces J&F by forcing empty masks; the net J&F effect is therefore determined by the balance between false-positive prevention and false-reject cost, which explains the modest raw J&F difference in Table 5.

## G. Refresh Trigger Analysis

Fig. 7 (left) decomposes refresh triggers into: pure EMAtriggered (event score), pure recurrence-triggered, and frames where both fire simultaneously. On DAVIS, 60.72% of triggers stem from the EMA event score alone, reflecting shorter videos with fewer re-appearance events. On MeViS and ReVOS, “Both” triggers grow to 45.08% and 50.82% respectively, consistent with longer and more dynamic videos. This decomposition directly answers the question of whether improvement comes mainly from the general event score or from re-appearance handling: both contribute, with recurrence handling becoming increasingly important in longer, more complex benchmarks, exactly matching the J&F patterns in Table 5. The SSIM distribution (right) further shows why SSIM-based baselines are threshold-sensitive: SSIM values cluster near 1.0 for all datasets, leaving a very narrow effective detection range.

## H. Pareto Trade-off Between Accuracy and Refresh Cost

Fig. 8 analyzes the accuracy-cost trade-off by plotting each configuration as a point in the plane of average #refresh (i.e., the average number of Sa2VA segmentation calls per video) versus J&F score. Since each refresh triggers an additional grounded segmentation inference, fewer refreshes directly translate to lower compute and latency. The dashed curve denotes the Pareto frontier, where no configuration can be improved in J&F without increasing refresh cost. Notably, many of our EDRRM configurations lie on or close to this frontier across Ref-DAVIS17 [5], MeViS [7], and ReVOS [8], indicating that adaptive, eventdriven boundaries achieve near-optimal accuracy for a given number of segmentation calls. This motivates our design choice: rather than refreshing at fixed intervals (Fixed Window) or relying on low-level appearance change heuristics (FrameDiff-SSIM), detecting object-composition change enables a segment sampler that concentrates Sa2VA calls at semantically meaningful moments, yielding a better accuracy-efficiency balance.

## I. Refresh Rate Budget and Budget-Conditioned J&F Efficiency

Fig. 9 and Table 8 summarize efficiency under a refresh-rate budget using fixed tuned settings. We instantiate each framework with a single best-tuned configuration from the ablations in Sec. IV-C: Fixed Window uses length L=8 (window-cfg-1), FrameDiff-SSIM uses threshold 0.93 (ssim-cfg-2), and EDRRM uses thr\_track=0.25 (edrrm-cfg-5). For a video instance v, we define its refresh rate as $r ( v ) = 1 0 0$ $N _ { \mathrm { r e f r e s h } } ( v ) / T ( v )$ , where $N _ { \mathrm { r e f r e s h } } ( v )$ is the number of refreshes (Sa2VA segmentation calls) and T(v) is the number of frames.

![](images/fe60bfd9fca67da1c845cd6a656922e8263385c726bd09f2d1df2a01ffb2de0c.jpg)

Table 5. Component Ablation. J&F (%) and average #refresh (#ref) for each variant on Ref-DAVIS17, MeViS, and ReVOS under the oracle-tuned best configuration. Each ablated row removes exactly one module from the full EDRRM system.
<table><tr><td>Variant</td><td>DAVIS [5] J&amp;F #ref</td><td>J&amp;F</td><td>MeViS [7] #ref</td><td>ReVOS [8] J&amp;F #ref</td></tr><tr><td>Sa2VA Original</td><td>75.20</td><td>1.00</td><td>57.00 1.00</td><td>57.60 1.00</td></tr><tr><td>EDRRM (Ours)</td><td>76.84</td><td>3.85</td><td>63.13 7.89</td><td>66.16 2.22</td></tr><tr><td>w/o recurrence memory</td><td>75.97</td><td>3.53</td><td>61.79 5.91</td><td>66.14 1.59</td></tr><tr><td>w/o identifiability gate</td><td>76.80</td><td>3.82</td><td>63.23 7.89</td><td>66.69 2.17</td></tr><tr><td>w/o anchor injection</td><td>76.24</td><td>3.82</td><td>62.77 7.22</td><td>66.14 2.22</td></tr><tr><td>w/o EMA</td><td>76.63</td><td>3.83</td><td>62.61 8.27</td><td>66.41 2.24</td></tr><tr><td>Event Score only</td><td>76.23</td><td>3.53</td><td>61.78 5.90</td><td>66.22 1.62</td></tr><tr><td>Recurrence only</td><td>76.68</td><td>3.20</td><td>62.42 5.30</td><td>66.15 2.01</td></tr></table>

Fig. 4. Component ablation Pareto frontier. J&F versus average #refresh for all ablation variants on Ref-DAVIS17 (a), MeViS (b), and ReVOS (c) The dashed curve marks the Pareto frontier. EDRRM (Ours) consistently lies on or near the frontier, confirming that the full system achieves the best accuracy-efficiency balance among all variants.  
Table 6. Tracker Sensitivity Analysis. Effect of thr\_track on J&F (%) and average #refresh (#ref). Higher thr\_track produces stricter, less noisy tracking with fewer but more reliable events.
<table><tr><td rowspan="2">thr_track</td><td rowspan="2">Tracking Quality</td><td colspan="2">DAVIS [5]</td><td colspan="2">MeViS [7]</td><td colspan="2">ReVOS [8]</td></tr><tr><td>J&amp;F</td><td>#ref</td><td>J&amp;F</td><td>#ref</td><td>J&amp;F</td><td>#ref</td></tr><tr><td>0.25</td><td>Very noisy</td><td>76.59</td><td>3.25</td><td>62.53</td><td>6.44</td><td>65.96</td><td>2.76</td></tr><tr><td>0.30</td><td>Noisy</td><td>76.53</td><td>3.21</td><td>61.99</td><td>6.82</td><td>65.57</td><td>2.49</td></tr><tr><td>0.42</td><td>Strict</td><td>76.76</td><td>1.78</td><td>61.21</td><td>4.08</td><td>65.34</td><td>1.02</td></tr><tr><td>0.50</td><td>Very Strict</td><td>76.78</td><td>1.25</td><td>61.61</td><td>2.46</td><td>65.34</td><td>1.05</td></tr></table>

Table 7. Identifiability Gate Statistics. Candidate refresh segments suppressed by the gate under the default EDRRM configuration (edrrm-cfg-5).
<table><tr><td>Dataset</td><td>Total Segments</td><td> $\overline { { \mathbf { G a t e } \ ^ { \mathbf { \omega } \bullet } \mathbf { n o } ^ { \mathbf { 3 } } } }$ </td><td>Gate &quot;no&quot; Rate</td></tr><tr><td>DAVIS</td><td>934</td><td>192</td><td>20.56%</td></tr><tr><td>MeViS</td><td>6224</td><td>1383</td><td>22.22%</td></tr><tr><td>ReVOS</td><td>6691</td><td>1713</td><td>25.60%</td></tr></table>

Given a budget $B \in [ 0 , 1 0 0 ]$ , we form (for each method) the subset of video instances whose refresh rate satisfies $r ( v ) < B$ To ensure a fair comparison at each budget, we evaluate all methods on the intersection of these subsets (i.e., the same set of instances shared across methods at that budget). The plotted value in Fig. 9 is then the budget-conditioned mean J&F over this common subset. These budget-conditioned values are computed on a budget-filtered shared subset and are therefore not directly comparable to full-dataset averages in Tables 2–4 and 6–7. Table 8 reports, for each dataset and method, the budget B<sup>⋆</sup> at which this budget-conditioned mean J&F is maximized, along with the corresponding peak J&F value. We also report segment-length statistics for the same fixed configurations; segment-length statistics are computed over the full benchmark under the same fixed configuration (not budget-filtered). Longer segment lengths are desirable since they imply fewer boundary splits (fewer segmentation calls) and more stable within-segment propagation.

Overall, our method reaches its peak J&F at smaller budgets than both baselines, meaning that high accuracy is achieved with fewer Sa2VA calls. This is practically important because a smaller refresh budget reduces the number of segmentation calls, lowering runtime and compute while maintaining strong RVOS accuracy.

![](images/19b8f95c542d791d947a48107cea2918eeff3941e778776f624f73d0af25841f.jpg)

![](images/595c9bed5a3acc96bd556e16fbf011baae3e6db243f7e48dced1de7510f34ff6.jpg)

![](images/1370fc58ec5737abfe4ac5fd936d62d99deda52ab9f6e733415ca522f5f42080.jpg)  
Fig. 5. J&F score versus integer refresh budget (5–30 Sa2VA calls). EDRRM, FrameDiff-SSIM, and Fixed-Window on Ref-DAVIS17, MeViS, and ReVOS. EDRRM maintains higher J&F across a broad budget range, particularly in MeViS, demonstrating that event-driven scheduling outperforms fixed-schedule and appearance-based baselines at matched inference cost.

![](images/64db2c2f21fbf42842746a45c17f789d79d9a1419202b2546be41ccf01cd1540.jpg)

![](images/92712135d1bd31572bd1672880b5f4f014d80ebe2c400bfb4c682cd64c6a259d.jpg)

![](images/b3ffb74a3a1da6982ece02e5dcdff2bef8570f3743dfb2fd8afe557258e125d6.jpg)  
Fig. 6. Tracker sensitivity analysis. J&F score (blue, left axis) and average #refresh (red dashed, right axis) versus thr\_track on Ref-DAVIS17 (a), MeViS (b), and ReVOS (c). Higher thr\_track reduces refresh count but also suppresses valid events, creating a graceful accuracy-cost trade-off.

Table 8. Comparison of refresh rate budget, segment length statistics, and J&F scores.
<table><tr><td rowspan="2">Method</td><td colspan="3">Refresh Rate Budget↓</td><td colspan="3">Segment Length (Avg) ↑</td><td colspan="3">Segment Length (Min/Max) ↑</td><td colspan="3">Peak budget-conditioned J&amp;F ↑</td></tr><tr><td>DAVIS</td><td>MEVIS</td><td>REVOS</td><td>DAVIS</td><td>MEVIS</td><td>REVOS</td><td>DAVIS</td><td>MEVIS</td><td>REVOS</td><td>DAVIS</td><td>MEVIS</td><td>REVOS</td></tr><tr><td>Window</td><td>15</td><td>14</td><td>17</td><td>7.58</td><td>7.52</td><td>6.72</td><td>4/8</td><td>4/8</td><td>3/7</td><td>78.78</td><td>64.03</td><td>66.38</td></tr><tr><td>SSIM</td><td>10</td><td>14</td><td>17</td><td>8.04</td><td>7.83</td><td>7.85</td><td>4/9</td><td>5/8</td><td>4/9</td><td>78.66</td><td>63.78</td><td>66.37</td></tr><tr><td>Ours</td><td>10</td><td>10</td><td>13</td><td>36.37</td><td>9.97</td><td>13.79</td><td>30/44</td><td>5/18</td><td>11/16</td><td>82.60</td><td>75.22</td><td>67.25</td></tr></table>

Table 9 further provides an apples-to-apples matched-budget comparison by grouping video instances into budget bins by average #calls $( \leq 2 , \leq 4 , \leq 8 )$ and reporting J&F only for videos within each bin, evaluated on the intersection of instances that all methods can serve at each budget level. A dash (–) indicates that no instances from that dataset fall within the bin for that method.

At budget ≤2, EDRRM is the only method that produces valid results across all three datasets (Window and SSIM have empty intersections on DAVIS and MeViS respectively at this tight budget), demonstrating that EDRRM’s event-driven scheduling operates at lower average call counts than fixedschedule alternatives. At budget ≤8, all methods are active and

EDRRM achieves the highest ReVOS score (66.15) among the three, confirming its advantage on the most dynamic benchmark even under strict budget constraints.

## J. End-to-End Runtime and Latency Analysis

To validate that the efficiency claim is supported by wallclock evidence, Table 10 reports per-component and end-toend average runtime (in seconds per video) across all three benchmarks. All measurements are taken on 4× NVIDIA RTX 8000 GPUs under the default EDRRM configuration $( \mathtt { e d r r m - c f g - 5 } )$

The results reveal three key findings. First, the dominant cost within EDRRM is Sa2VA inference (21.66/24.93/9.26 s) and MASA tracking (18.73/18.97/8.46 s); together they account for over 90% of total runtime. In contrast, the lightweight schedul ing components (Event Score, CLIP Recurrence, Identifiability Gate) add only 3.09/7.65/2.15 s in overhead, which is modest relative to the segmentation cost. Second, EDRRM total runtime exceeds Sa2VA Original because it invokes Sa2VA multiple times per video (avg 3–8 calls vs. a single call for Original); this overhead is the price of reduced stale grounding and improved accuracy. Third, FrameDiff-SSIM has unexpectedly high MeViS latency (41.37 s) due to its high frame-difference computation cost on long MeViS videos, while EDRRM’s total (51.55 s) reflects its higher refresh count on that benchmark. These results confirm that EDRRM’s computational overhead is well-characterized and predictable, with the scheduling modules themselves contributing negligible cost.

![](images/f8024215987963c23ce56c641279ebd9b1e1296cb0179386a3cc1d8d6f206eef.jpg)

Fig. 7. Refresh trigger analysis. Left: Proportion of refresh triggers attributed to EMA event score only (blue), recurrence matching only (red), or both simultaneously (purple) per dataset. Right: SSIM value distribution across all consecutive frame pairs in each benchmark, illustrating why SSIM-based detection is threshold-sensitive.  
![](images/8df94dc858c326f5e7a40ebb9a89aa0ffb5954f8a982c8dfacc241feadc5c38f.jpg)  
Fig. 8. J&F score versus average #refresh (Sa2VA segmentation calls) for all configurations.

## K. Main Results

Tables 11 and 12 report the primary RVOS results on Ref-DAVIS17 [5], MeViS [7], and ReVOS [8] under two tuning protocols. Table 11 (global-tuned) uses a single fixed configuration per method across all datasets: Fixed Window uses window-cfg-1 (L=8), FrameDiff-SSIM uses ssim-cfg-2 (thr= 0.93), and our EDRRM uses edrrm-cfg-5. Under this setting, Sa2VA-8B + EDRRM achieves the best overall mean score (68.49), improving over Sa2VA-8B + fixed (68.45) and Sa2VA-8B + ssim (68.39), while also yielding the strongest MeViS and ReVOS scores (63.13 and 65.75). Table 12 (oracle per-dataset tuned) allows each method to choose its best configuration separately for each dataset (i.e., best L for Fixed Window, best SSIM threshold for FrameDiff-SSIM, and best EDRRM setting per dataset). Even under this more favorable tuning, Sa2VA-8B + EDRRM remains the best overall, achieving the highest mean (68.71) and the best scores on MeViS (63.13) and ReVOS (66.16), with Ref-DAVIS17 also competitive (76.84). These results indicate that selectively refreshing grounding at event-driven change points and leveraging recurrence anchors provides a consistent accuracy-efficiency advantage: while the absolute J&F gains in the full-dataset comparison are modest, the improvement is most pronounced under budget-constrained settings (Sec. IV-I,

![](images/9a23591d489a9ba744847cfe78157a313371dd1f52a9567ea8645f6963bc7a71.jpg)

![](images/fea9194862b09dfab0c321a7922b6595def826c03504fa80fd3e78393aa2521d.jpg)

![](images/857c80c793f4cd72a357633ffab49d26d0f0026269ea58599c5dc75866c8418e.jpg)  
Fig. 9. Budget-conditioned mean J&F versus refresh-rate budget. Note: Budgets where this shared intersection is empty yield undefined means; we therefore report the non-empty range (10–22%) in our benchmarks.

Table 9. Matched Budget Comparison. J&F (%) evaluated on the shared intersection of video instances where each method uses at most the stated average number of Sa2VA calls. A dash (–) indicates an empty intersection for that dataset at that budget. Note that at budget <=2, the qualifying subset is very smal and consists predominantly of static or easy videos, so per-method scores at this level are not indicative of general performance.
<table><tr><td>Budget (avg #calls)</td><td>Method</td><td>DAVIS [5]</td><td>MeViS [7]</td><td>ReVOS [8]</td></tr><tr><td rowspan="3">≤2</td><td>Window</td><td></td><td></td><td>67.37</td></tr><tr><td>SSIM</td><td>93.64</td><td></td><td>64.71</td></tr><tr><td>EDRRM</td><td>77.55</td><td>66.08</td><td>67.06</td></tr><tr><td rowspan="3">≤4</td><td>Window</td><td></td><td>63.50</td><td>66.03</td></tr><tr><td>SSIM</td><td>78.66</td><td></td><td>66.55</td></tr><tr><td>EDRRM</td><td>78.49</td><td>58.62</td><td>66.30</td></tr><tr><td rowspan="3">≤8</td><td>Window</td><td>81.36</td><td>61.81</td><td>65.89</td></tr><tr><td>SSIM</td><td>81.00</td><td>63.25</td><td>65.61</td></tr><tr><td>EDRRM</td><td>77.68</td><td>61.86</td><td>66.15</td></tr></table>

Table 10. End-to-End Runtime / Latency Breakdown (seconds per video). Per-component latency for EDRRM and total end-to-end latency compared against Sa2VA Original, Fixed-window, and FrameDiff-SSIM baselines.
<table><tr><td>Component</td><td>DAVIS [5]</td><td>MeViS [7]</td><td>ReVOS [8]</td></tr><tr><td>MASA Tracking</td><td>18.73</td><td>18.97</td><td>8.46</td></tr><tr><td>Event Score</td><td>0.18</td><td>0.64</td><td>0.19</td></tr><tr><td>CLIP Recurrence</td><td>0.17</td><td>0.63</td><td>0.18</td></tr><tr><td>Identifiability Gate</td><td>2.74</td><td>6.38</td><td>1.78</td></tr><tr><td>Sa2VA Inference</td><td>21.66</td><td>24.93</td><td>9.26</td></tr><tr><td>EDRRM Total</td><td>43.48</td><td>51.55</td><td>19.87</td></tr><tr><td>Sa2VA Original</td><td>11.41</td><td>11.67</td><td>5.93</td></tr><tr><td>Fixed-window</td><td>24.82</td><td>25.46</td><td>9.78</td></tr><tr><td>FrameDiff-SSIM</td><td>25.75</td><td>41.37</td><td>15.15</td></tr></table>

Table 9) and in the more dynamic benchmarks (MeViS, ReVOS) where stale grounding failures are more frequent. The results should therefore be interpreted primarily as a demonstration of improved accuracy at matched or lower refresh cost, rather than as large absolute accuracy gains.

## L. Qualitative Comparison

Fig. 10 presents a representative failure case of vanilla Sa2VA-8B and the corresponding improvement from our event-driven refresh. Given the query “Please segment the white sedan car ahead in my lane,” the vanilla model produces a persistent false positive: it locks onto an incorrect instance early and continues to propagate this mistaken mask throughout the video. This behavior is consistent with Sa2VA’s inference design, where the MLLM is conditioned on a small fixed set of initial keyframes (e.g., the first few frames) to decide the target grounding, after which segmentation is largely driven by propagation; if the initial grounding is imperfect or the scene composition changes later, the model lacks a mechanism to re-ground and correct the identity. In contrast, our Event Refresh triggers re-invocation at detected object-composition change points, re-conditioning the model on updated frames and thereby suppressing false positives and maintaining the correct target mask over time. This example highlights the motivation for an adaptive segment sampler: by refreshing only when meaningful changes occur, the pipeline can recover from stale grounding while avoiding unnecessary segmentation calls.

Table 11. Method/Model comparison: Global-tuned (single setting) across all datasets.
<table><tr><td>Method/Model</td><td>Tuning</td><td>DAVIS [5]</td><td>MEVIS [7]</td><td>REVOS [8]</td><td>Mean</td></tr><tr><td>VideoLISA-3.8B</td><td>-</td><td>68.80</td><td>44.40</td><td></td><td>56.60</td></tr><tr><td>VISA-13B</td><td></td><td>70.40</td><td>44.50</td><td>50.90</td><td>55.27</td></tr><tr><td>Sa2VA-1B</td><td></td><td>72.30</td><td>50.80</td><td>47.60</td><td>56.90</td></tr><tr><td>Sa2VA-4B</td><td></td><td>73.80</td><td>52.10</td><td>53.20</td><td>59.70</td></tr><tr><td>Sa2VA-8B</td><td></td><td>75.20</td><td>57.00</td><td>57.60</td><td>63.27</td></tr><tr><td>Sa2VA-26B</td><td></td><td>77.00</td><td>57.30</td><td>58.40</td><td>64.23</td></tr><tr><td>Sa2VA-8B + fixed</td><td>window-cfg-1 (L=8)</td><td>77.45</td><td>62.34</td><td>65.55</td><td>68.45</td></tr><tr><td>Sa2VA-8B + ssim</td><td>ssim-cfg-2 (thr=0.93)</td><td>77.39</td><td>62.24</td><td>65.55</td><td>68.39</td></tr><tr><td>Sa2VA-8B + EDRRM</td><td>edrrm-cfg-5</td><td>76.58</td><td>63.13</td><td>65.75</td><td>68.49</td></tr></table>

Prompt: "Please segment the white sedan car ahead in my lane"  
![](images/3dd13d5341813437229ab7c926b9e2447ad60a3673c1f7b4b475edae78a67381.jpg)

![](images/80fb73d284a2cfb8a50b9e56c0b9cfdacd97cfdf14210c13a5d1184eee7c05ff.jpg)  
Fig. 10. Qualitative Comparison.

Table 12. Method/Model comparison: Best-case per-dataset tuned (oracle configuration).
<table><tr><td>Method/Model</td><td>Tuning</td><td>DAVIS [5]</td><td>MEVIS [7]</td><td>REVOS [8]</td><td>Mean</td></tr><tr><td>VideoLISA-3.8B</td><td>-</td><td>68.80</td><td>44.40</td><td></td><td>56.60</td></tr><tr><td>VISA-13B</td><td></td><td>70.40</td><td>44.50</td><td>50.90</td><td>55.27</td></tr><tr><td>Sa2VA-1B</td><td></td><td>72.30</td><td>50.80</td><td>47.60</td><td>56.90</td></tr><tr><td>Sa2VA-4B</td><td></td><td>73.80</td><td>52.10</td><td>53.20</td><td>59.70</td></tr><tr><td>Sa2VA-8B</td><td></td><td>75.20</td><td>57.00</td><td>57.60</td><td>63.27</td></tr><tr><td>Sa2VA-26B</td><td></td><td>77.00</td><td>57.30</td><td>58.40</td><td>64.23</td></tr><tr><td>Sa2VA-8B + fixed</td><td>best L per dataset</td><td>77.45</td><td>62.34</td><td>65.88</td><td>68.56</td></tr><tr><td> $\mathrm { S a 2 V 4  – 8 B + s s i m }$ </td><td>best SSIM per dataset</td><td>77.39</td><td>62.48</td><td>65.55</td><td>68.47</td></tr><tr><td> $\mathrm { S a 2 V A  – 8 B + E D R R M }$ </td><td>best tune per dataset</td><td>76.84</td><td>63.13</td><td>66.16</td><td>68.71</td></tr></table>

## V. CONCLUSION AND FUTURE WORK

We presented EDRRM, an event-driven refresh and recurrence-aware enhancement for Sa2VA [1] that adaptively schedules grounded segmentation calls based on objectcomposition change and re-identification cues. Across Ref-DAVIS17 [5], MeViS [7], and ReVOS [8], our analyses show that EDRRM produces interpretable refresh boundaries (Fig. 3), yields configurations that lie close to the Pareto frontier of J&F versus segmentation-call cost (Fig. 8), and achieves higher accuracy under smaller refresh-rate budgets (Figs. 5, 9, Tables 8 and 9). Furthermore, component ablations confirm that the full system dominates all ablated variants on the same frontier (Fig. 4), highlighting a meaningful accuracy-efficiency tradeoff. The absolute J&F gains in the full-dataset setting are modest; the primary advantage of EDRRM is demonstrated in budget-constrained conditions and in the more dynamic MeViS and ReVOS benchmarks where event-driven refresh recovers re-appearance failures that fixed-schedule methods miss. In the main quantitative comparisons, EDRRM achieves competitive or superior overall mean scores under both globaltuned and oracle-tuned protocols (Tables 11 and 12), while qualitative results confirm that refresh can mitigate persistent false positives caused by stale initial grounding (Fig. 10).

End-to-end runtime analysis (Table 10) confirms that EDRRM’s scheduling components (event score, CLIP recurrence, identifiability gate) add modest overhead relative to the dominant Sa2VA inference and MASA tracking costs. Component ablations (Table 5, Fig. 4) confirm that both the event score and recurrence memory contribute complementarily to the accuracy-efficiency balance. Tracker sensitivity analysis (Table 6) demonstrates graceful degradation under noisy tracking, with J&F varying by at most 1.3 points across the full thr\_track sweep, and gate statistics (Table 7) confirm that the identifiability gate actively suppresses 20–26% of false-positive refresh calls across benchmarks.

Despite these gains, our method introduces additional hyperparameters (e.g., event thresholds, cooldowns, and recurrence similarity/TTL) and relies on tracking quality for robust event estimation; failures in detection/association may delay refresh or trigger unnecessary boundaries. The event score weights are currently hand-tuned rather than learned, and CLIP-based recurrence matching remains susceptible to failure under heavy occlusion or same-class distractors, though the identifiability gate and label gating substantially mitigate these risks as shown in Sec. IV-F. Recurrence matching is also sensitive to embedding quality under viewpoint change or low resolution, and computing image embeddings can add overhead in highrefresh regimes. Future work includes learning the event trigger from data to reduce manual tuning, incorporating stronger appearance/motion features for recurrence matching under challenging conditions, and extending the scheduler to streaming/online RVOS with explicit latency constraints. We also plan to explore tighter integration between refresh decisions and mask-propagation confidence to further suppress false positives while minimizing segmentation calls.

## ACKNOWLEDGMENTS

This work was partly supported by Center for Applied Research in Artificial Intelligence (CARAI) grant funded by

DAPA and ADD (UD230017TD) and partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-24742969, Intelligent Robotic System using Continual Learning and Multimodal Language Model based Multi Attribute Feedback).

[23] Z. Wu, C. Xiong, C.-Y. Ma, R. Socher, and L. S. Davis, “AdaFrame: Adaptive Frame Selection for Fast Video Recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019.

[24] A. Ghodrati, B. E. Bejnordi, and A. Habibian, “FrameExit: Conditional Early Exiting for Efficient Video Recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021.

## REFERENCES

[1] Z. Yuan et al., “Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos,” arXiv preprint arXiv:2501.04001, 2025.

[2] N. Ravi et al., “SAM 2: Segment Anything in Images and Videos,” arXiv preprint arXiv:2408.00714, 2024.

[3] A. Kirillov et al., “Segment Anything,” arXiv preprint arXiv:2304.02643, 2023.

[4] A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” in Proc. Int. Conf. Mach. Learn. (ICML), 2021. (arXiv:2103.00020)

[5] A. Khoreva, A. Rohrbach, and B. Schiele, “Video Object Segmentation with Referring Expressions,” in Asian Conference on Computer Vision (ACCV), 2018.

[6] S. Seo et al., “Unified Referring Video Object Segmentation Network with a Large-Scale Synthetic Dataset,” in Proc. European Conf. on Computer Vision (ECCV), 2020.

[7] H. Ding, C. Liu, S. He, X. Jiang, and C. C. Loy, “MeViS: A Large-scale Benchmark for Video Segmentation with Motion Expressions,” in Proc. IEEE/CVF Int. Conf. on Computer Vision (ICCV), 2023.

[8] C. Yan et al., “VISA: Reasoning Video Object Segmentation via Large Language Models,” in Proc. European Conf. on Computer Vision (ECCV), 2024.

[9] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” IEEE Trans. Image Processing, vol. 13, no. 4, pp. 600–612, 2004.

[10] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual Instruction Tuning,” in Advances in Neural Information Processing Systems (NeurIPS), 2023. (arXiv:2304.08485)

[11] M. Bellver, C. Ventura, C. Silberer, I. Kazakos, J. Torres, and X. Giró- i-Nieto, “RefVOS: A Closer Look at Referring Expressions for Video Object Segmentation,” arXiv preprint arXiv:2010.00263, 2020.

[12] J. Wu, Y. Jiang, P. Sun, Z. Yuan, and P. Luo, “Language as Queries for Referring Video Object Segmentation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022.

[13] A. Botach, E. Zheltonozhskii, and C. Baskin, “End-to-End Referring Video Object Segmentation with Multimodal Transformers,” arXiv preprint arXiv:2111.14821, 2021.

[14] D. Wu, T. Wang, Y. Zhang, X. Zhang, and J. Shen, “OnlineRefer: A Simple Online Baseline for Referring Video Object Segmentation,” arXiv preprint arXiv:2307.09356, 2023.

[15] Z. Luo, Y. Xiao, Y. Liu, S. Li, Y. Wang, Y. Tang, X. Li, and Y. Yang, “SOC: Semantic-Assisted Object Cluster for Referring Video Object Segmentation,” in Advances in Neural Information Processing Systems (NeurIPS), 2023. (arXiv:2305.17011)

[16] K. Gavrilyuk, A. Ghodrati, Z. Li, and C. G. M. Snoek, “Actor and Action Video Segmentation from a Sentence,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018. (arXiv:1803.07485)

[17] J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbelaez, A. Sorkine-Hornung, and L. Van Gool, “The 2017 DAVIS Challenge on Video Object Segmentation,” arXiv preprint arXiv:1704.00675, 2017.

[18] H. K. Cheng, Y.-W. Tai, and C.-K. Tang, “Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation,” arXiv preprint arXiv:2106.07452, 2021.

[19] H. K. Cheng et al., “XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model,” arXiv preprint arXiv:2207.07115, 2022.

[20] Z. Yang, Y. Wei, and Y. Yang, “Associating Objects with Transformers for Video Object Segmentation,” in Advances in Neural Information Processing Systems (NeurIPS), 2021. (arXiv:2106.02638)

[21] S. Li, L. Ke, M. Danelljan, L. Piccinelli, M. Segù, L. Van Gool, and F. Yu, “Matching Anything by Segmenting Anything,” arXiv preprint arXiv:2406.04221, 2024.

[22] X. Li, J. Wang, X. Xu, X. Li, B. Raj, and Y. Lu, “Robust Referring Video Object Segmentation with Cyclic Structural Consensus,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023. (arXiv:2207.01203)