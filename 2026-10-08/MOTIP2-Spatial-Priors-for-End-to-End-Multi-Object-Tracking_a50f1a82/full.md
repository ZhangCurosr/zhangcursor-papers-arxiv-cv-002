# MOTIP2: Spatial Priors for End-to-End Multi-Object Tracking

Benoît Roussel<sup>1,2</sup> benoit.roussel@ec-lyon.fr

Damien Bouet<sup>2</sup> 6damien.bouet@idemia.com

<sup>2</sup>Liming Chen<sup>1</sup> liming.chen@ec-lyon.fr

Pierre Perrault<sup>2</sup> <sup>c</sup>pierre.perrault@idemia.com

<sup>1</sup> École Centrale de Lyon LIRIS, UMR CNRS 5205 Lyon, France

<sup>2</sup>IDEMIA Courbevoie, France

## Abstract

End-to-end multi-object trackers have narrowed the gap with classical tracking-bydetection on association-difficult benchmarks. Yet they still make spatially implausible errors no classical tracker would, such as assigning one identity to objects on opposite sides of the frame. A model could learn to avoid them, but tracking annotations are scarce, so we encode spatial priors explicitly instead, while keeping inference fully end-to-end with no post-hoc association.

We propose three spatial priors, at the data, loss, and representation stages. Spatial ID Switches bias trajectory permutations toward spatially overlapping objects, reducing the mismatch between training and inference confusions. Spatial ID Loss scales each identity’s penalty by its box distance, so a distant switch costs more than a nearby one. Spatial Anchor gives each track token its frame position, an explicit spatial cue for attention.

We instantiate the three priors in MOTIP2, a tracker adapted from MOTIP and built on the real-time DEIM detection transformer. Trained without extra data, its main model, MOTIP2-L, sets a new state of the art: 73.4 HOTA on DanceTrack, 76.0 on SportsMOT, and 71.1 IDF1 on PersonPath22. MOTIP2 is a family of models spanning the speedaccuracy trade-off: a lighter model, MOTIP2-S, matches the original MOTIP at over 3× the speed, and MOTIP2-X reaches 74.8 HOTA on DanceTrack.

## Introduction

Multi-object tracking supports autonomous driving [53], sports analytics [7], and video surveillance [41], all of which need stable identities through occlusion and crowding. Endto-end multi-object trackers [10, 29, 54] extend the DETR [5] query paradigm to tracking: detection without NMS, and learned association replacing the hand-crafted IoU matching that breaks down under heavy occlusion. They match or surpass classical tracking-bydetection [1, 3, 4, 58] on association-difficult benchmarks such as DanceTrack [45].

![](images/7e03c261d26ac92aabde34e9f4d8091b06013a0c82c00e0176720b1a2b4cad01.jpg)  
Figure 1: Accuracy vs. latency. HOTA on DanceTrack-val vs. per-frame PyTorch FP32 latency (ms) on a single V100. At matched accuracy, MOTIP2-S is 3.2× faster than MOTIP. Marker shape denotes input resolution: square 640×640, gray triangle 800×1333; MOTIP (black) and MOTIP2 (circles) run at $8 0 0 \times 1 4 4 0$ <sup>†</sup> Estimated from RTX 4090 measurements [19] scaled by MOTIP’s V100/4090 ratio.

Two recurring failure types, however, share one cause: association without spatial grounding. In the first, two distant boxes in consecutive frames, with no occlusion between them, are assigned the same identity (see Appendix D), a failure rare in classical trackers, whose motion gating makes such jumps expensive. In the second, the tracker swaps identities when objects cross or occlude. Neither is constrained by geometry: the cross-entropy penalizes an implausible cross-frame jump no more than an adjacent mistake, and training barely exposes the model to the nearby confusions where association is genuinely ambiguous. The association step needs an explicit spatial prior.

Tracking-by-detection encodes such constraints for free, as hand-crafted heuristics that need no training data: Kalman motion, IoU cost, and overlap-gated re-identification each forbid an association the moment it becomes spatially implausible [1, 48]. End-to-end trackers discard these components and leave the attention mechanism to learn the same constraints from data. But annotated tracking data is scarce, so these constraints are learned imperfectly, and the tracker makes the implausible associations they would have prevented.

We propose three explicit spatial priors to tackle both failures, one each at the data, loss, and representation stage, all encoding one bias: spatial proximity should govern association. At the data stage, Spatial ID Switches target the dominant occlusion confusions. MOTIP exposes its decoder to such errors by shuffling historical identities at random, but a random shuffle mostly fabricates long-range swaps that rarely occur at inference, while barely sampling the nearby occlusions that cause most errors. We instead bias permutations toward spatially overlapping objects, so augmented histories mirror the confusions the tracker actually encounters. This trains the decoder to look further back in the tracklet to disambiguate when nearby objects are present, rather than blindly trusting the most recent assignment (Fig. 4).

At the loss stage, the Spatial ID Loss targets the distant jump by restoring the geometric cost the cross-entropy leaves out. At the representation stage, the Spatial Anchor ties each track token to its frame position, a dedicated channel through which geometry enters association, so the model discriminates between candidates by where they are rather than only penalizing incoherent associations after the fact. Unlike the heuristic stack, these priors add no handcrafted matching at test time, leaving inference free of any post-hoc association module.

We instantiate the three priors in MOTIP2, an end-to-end tracker built on MOTIP [10] with a real-time DEIM [13] detection backbone and an adapted ID decoder, and evaluate it on DanceTrack [45], SportsMOT [7], and PersonPath22 [41]. The gains separate cleanly: the DEIM backbone drives detection and speed, while under a fixed backbone the priors alone add up to +2.1 HOTA and +3.5 AssA on DanceTrack test (Tab. 1). Transformer trackers have long been latency-bound; MOTIP2 runs at real-time rates, over 3× faster than MOTIP at matched accuracy (Fig. 1).

## Contributions.

(i) Spatial ID Switches: a data-stage augmentation that replaces uniform tracklet shuffling with overlap-weighted permutations.

(ii) Spatial ID Loss: a loss-stage term that scales each candidate identity’s penalty by its box distance to the detection.

(iii) Spatial Anchor: a representation-stage change that reinjects the detector’s positional encoding into the ID decoder, reusing a spatial cue it already computes.

(iv) MOTIP2: a real-time, DEIM-based end-to-end tracker, adapted from MOTIP, that validates the three priors with consistent gains across three tracking benchmarks.

## 2 Related Work

Tracking-by-detection. Tracking-by-detection pipelines [1, 3, 4, 48, 58] pair a frame-level detector [11] with a small, hand-designed association stack: a Kalman filter [15] propagates each track forward, IoU cost penalizes implausible jumps, and Hungarian matching [16] resolves the assignment. When appearance is added via a re-identification network [1, 47, 48, 50, 57], its score is gated by motion or overlap rather than consulted in isolation. Every component carries an explicit spatial prior, and the association module itself has essentially no learned parameters; this kept the pipeline competitive while annotated tracking data stayed scarce. These pipelines also commonly delay committing tentative tracks until they are observed across several consecutive frames [48], a tracklet-confirmation mechanism that prior query-based trackers have not adopted. The stack fails when crowding makes these geometric priors ambiguous: motion gating and IoU cost both lose their discriminative power as the IoU between an object’s consecutive observations collapses toward the inter-object IoU with its neighbors. This limitation motivated learned association.

Detection Transformers. DETR [5] reformulates detection as set prediction on the transformer [46], dispensing with the NMS step that suppresses valid detections when targets overlap. A line of follow-up work progressively re-grounds the decoder in geometry: Deformable DETR [63] attaches reference points to each query, DAB-DETR [23] represents queries explicitly as 4D anchor boxes, and DN-DETR [17] and DINO [55] add box denoising objectives that anchor queries against ground-truth geometry during training. Real-time variants [13, 33, 60] bring this family to tracking-by-detection latencies while retaining the anchored-query structure; we build on DEIM [13] for its accuracy at real-time latency.

Tracking-by-query. Transformer-based trackers [29, 44, 54] inherit DETR’s set-prediction formulation and add temporal association by propagating object queries across frames. MOTR [54] formalizes this as a fully end-to-end framework, but the decoder it shares between detection and association creates a supervision conflict that MOTRv2 [59], MOTRv3 [52], and CO-MOT [49] address through query-specific supervision or external proposals. MOTIP [10] resolves the conflict architecturally: a dedicated ID decoder predicts identity labels from the detector’s outputs, decoupling the two tasks and enabling scalable training, with trajectorylevel identity permutations as the main association augmentation. The ID decoder, however, consumes only the detector’s content embeddings: the explicit per-query anchor that DAB-DETR and its successors introduced inside the detector is not carried through to association, leaving the module to recover spatial coherence from content alone. FDTA [40], also building on MOTIP, instead learns more discriminative object embeddings for association and is our primary baseline (Sec. 4.2). Transformer trackers also remain substantially slower than tracking-by-detection, a gap that FastTrackTr [19] begins to close. Concurrent work raises end-to-end association accuracy through segmentation-driven tracking built on foundation models [14], a direction complementary to the spatial priors we introduce. Our work targets both gaps, spatial grounding and speed: we restore the missing spatial prior at three stages of the association pipeline (data, loss, representation) on a real-time DEIM backbone.

## 3 Spatial Priors for ID Association

We first present MOTIP2, our end-to-end tracker adapted from MOTIP [10], together with modifications to its detector and ID decoder (Sec. 3.1, Fig. 2). We then instantiate the three spatial priors announced in Sec. 1, each acting at a different stage of the tracker: Spatial ID Switches at the data stage (Sec. 3.2), the Spatial ID Loss at the loss stage (Sec. 3.3), and Spatial Anchor at the representation stage (Sec. 3.4).

## 3.1 Overview: MOTIP2

MOTIP2 follows MOTIP [10], which we briefly recap; readers familiar with it can skip to the changes. MOTIP decouples detection from identity prediction within a single end-to-end tracker. A detection transformer processes each frame independently, producing detection embeddings $\mathbf { e } _ { i }$ with bounding boxes $\hat { b } _ { i }$ . During training, bipartite matching assigns each prediction to a ground-truth object with identity label $y _ { i }$ , as in DETR; a dedicated ID decoder then cross-attends over a historical context of the last T frames and predicts, for each detection, a distribution over an ID dictionary of K+1 learnable identity embeddings (K active identities plus a newborn class). The model is trained end-to-end on short clips sampled at random frame rates, jointly optimizing the detector and the ID decoder. At inference the video is processed online: each frame’s assigned embeddings are appended to the context, maintained as a FIFO buffer of size T, and a greedy protocol turns the predicted distributions into unique identity assignments.

Throughout, end-to-end means no post-hoc association module rather than no tracking logic at inference: the greedy readout and the tracklet confirmation below manage track lifecycles, with no matching stage between separately produced detections and tracks. Beyond the three spatial priors that motivate this work (Secs. 3.2 to 3.4), our changes are confined to the detector, the ID decoder, and the training objective, described next; everything else above is inherited unchanged.

![](images/c3b02bd6baa2405cdbe26730514ca0961f46b2c1787e83a7435db0bca6207381.jpg)  
Figure 2: Overview of MOTIP2. DEIM produces detection embeddings independently per frame. Three explicit spatial priors then inject geometric structure at distinct stages: (i) Spatial ID Switches perturb the training-time historical context by GIoU-biased identity permutations that mirror the near-track confusions seen at inference; (ii) Spatial ID Loss supervises the ID decoder with a geometric penalty; (iii) Spatial Anchor carries each detection’s positional encoding into the ID decoder, so every token holds an explicit cue to its image position.

Detection backbone. We replace MOTIP’s Deformable DETR detector [63] with the realtime detector DEIM [13]. Profiling shows that over 80% of MOTIP’s inference cost comes from the detector (Tab. 4), making it the primary target for acceleration. Selecting different DEIM backbone and decoder configurations instantiates a family of models (N/S/M/L/X): even the high-capacity X variant matches MOTIP’s speed while being more accurate, and lighter variants cut latency further.

ID decoder. We keep MOTIP’s ID decoder with two structural changes. First, we fuse detection features and identity embeddings by element-wise addition rather than concatenation, halving the hidden dimension to match the DETR decoder; we double the feed-forward network (FFN) ratio to preserve capacity. Second, we replace MOTIP’s learned discrete temporal attention bias with rotary position embeddings (RoPE) [43] applied inside the query–key product, so attention between two tokens varies with their frame offset. RoPE enables context-window extension at inference beyond the training length [6]; because training samples clips at variable frame rates while inference runs at a fixed stride, we rescale the RoPE timescale at inference by a factor α to match the average training timescale. Finally, at inference we add a lightweight tracklet-confirmation step absent from MOTIP: a track is confirmed only after $N _ { \mathrm { i n i t } } { = } 2$ consecutive consistent assignments, and unconfirmed tracks are discarded with their dictionary slots freed.

## 3.2 Spatial ID Switches

Because bipartite matching assigns each token to its ground-truth identity, the training-time historical context is, by construction, perfectly consistent. At inference, however, noisy detections and occlusions cause identity ambiguities. To bridge this gap, MOTIP [10] perturbs the training context with an ID-switch augmentation that selects a random subset of objects in each frame and permutes their identities among themselves. The ID decoder thus learns to recover from confusions rather than overfitting to a perfect history.

![](images/ab79519cc4adbe4dca94b03d7fc9db3c77d1ceafe70c1e2e8c4230eb2ec68d2a.jpg)  
Figure 3: Spatial ID Switches. A DanceTrack [45] frame with four annotated objects, of which only 2 and 3 overlap, and the identity permutation the augmentation draws for this frame. We perturb the pairwise GIoU affinity with i.i.d. Gaussian noise $\varepsilon _ { i j } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ and let the Hungarian algorithm return the assignment maximizing it; cell $( i , j )$ means object i takes identity j, light where the identity is kept and dark where it is swapped. Since only spatially close pairs carry positive affinity, the perturbation swaps ${ 2  3 }$ while the well-separated objects keep their identity.

Genuine confusions, however, are local: they arise between nearby objects under occlusion or close passage. Objects on opposite sides of the frame are never genuinely ambiguous, yet a uniform permutation, blind to geometry, fabricates mostly these long-range pairings while under-sampling the nearby confusions that cause most errors.

We keep this random selection of objects and change only how the selected identities are reassigned, biasing the permutation toward spatially proximate boxes instead of drawing it uniformly. Given the selected boxes in frame t, we compute their pairwise GIoU [35] affinity matrix $G _ { t }$ and perturb it with Gaussian noise:

$$
\begin{array} { r } { A _ { t } = G _ { t } + \varepsilon , \quad \varepsilon _ { i j } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma ^ { 2 } ) , } \end{array}
$$

where $\sigma > 0$ sets the GIoU scale below which two objects are treated as confusable. The augmented permutation on the random subset is then obtained by solving the linear assignment on $A _ { t }$ via the Hungarian algorithm (Fig. 3).

The augmentation recovers $\mathrm { M O T I P ^ { \circ } s }$ uniform permutation exactly as $\sigma \to \infty$ , where noise dominates the affinity. For finite $\sigma .$ , an object swaps only when a nearby box’s perturbed affinity overtakes its self-affinity $( G _ { i i } { = } 1 )$ , so $\sigma$ is a confusion radius in GIoU space: objects with a close, high-GIoU neighbor face a real swap probability while well-separated objects stay virtually untouched. The augmentation thus adapts to scene geometry on its own, and because locality is fixed by geometry rather than a tuned swap rate, we use a single $\sigma { = } 0 . 5$ on all benchmarks without per-dataset tuning.

## 3.3 Spatial ID Loss

Cross-entropy $( \mathcal { L } _ { \mathrm { C E } } )$ over the ID softmax is spatially blind: a confident wrong prediction on a track across the frame is penalized identically to a confusion with an adjacent track. The gradient carries no geometric signal. We complement it with two spatial terms (L1 and GIoU) that weight each candidate identity by the distance between the detection and that identity’s box:

$$
\mathcal { L } _ { \mathrm { i d - 1 1 } } = \sum _ { c } p _ { c } \| \hat { b } - b _ { c } ^ { \mathrm { g t } } \| _ { 1 } , \qquad \mathcal { L } _ { \mathrm { i d - g i o u } } = \sum _ { c } p _ { c } \left( 1 - \mathrm { G I o U } ( \hat { b } , b _ { c } ^ { \mathrm { g t } } ) \right) ,\tag{1}
$$

where $p _ { c }$ is the softmax probability for identity $c , \hat { b }$ is the predicted box of the detection being classified, and $b _ { c } ^ { \mathrm { g t } }$ is the ground-truth box of the object holding identity $^ { c , }$ , taken from the same bipartite matching that supervises detection (Sec. 3.1). The sum runs over the identities present in the current frame plus the newborn class: identities absent from the frame have no box, so the probabilities $p _ { c }$ are renormalized over this reduced set rather than taken from the full-dictionary softmax. The newborn class has no associated object, so we set $b _ { c } ^ { \mathrm { g t } }$ to the detection’s own ground-truth box, reducing the term to its localization error and leaving the cross-entropy to separate newborn from the correct existing identity. Because $\hat { b }$ is detached, these terms shape only the ID logits and leave the detection regression head untouched.

The total ID objective adds these two spatial terms to the cross-entropy, weighted by the same $( \lambda _ { \mathrm { C E } } , \lambda _ { \mathrm { G I o U } } , \lambda _ { \mathrm { L 1 } } )$ as the DEIM detection loss:

$$
\mathcal { L } _ { \mathrm { i d } } = \lambda _ { \mathrm { C E } } \mathcal { L } _ { \mathrm { C E } } + \lambda _ { \mathrm { G I o U } } \mathcal { L } _ { \mathrm { i d - g i o u } } + \lambda _ { \mathrm { L 1 } } \mathcal { L } _ { \mathrm { i d - l 1 } } .\tag{2}
$$

## 3.4 Spatial Anchor

Detection in modern DETR variants [13, 17, 23, 55, 60] is built on explicit spatial grounding: each query is a 4D anchor box, refined layer by layer, and a positional encoding $\mathbf { p } _ { i }$ (a learned embedding of the final-layer box) steers cross-attention toward the relevant image region at every decoder step. At the handoff to the ID decoder, however, only the content embedding $\mathbf { e } _ { i }$ is carried forward; $\mathbf { p } _ { i }$ is dropped, leaving the decoder to perform association without any explicit cue about where on the frame each track sits, despite this information being available one stage upstream.

End-to-end training could let $\mathbf { e } _ { i }$ encode location implicitly, but only entangled with appearance and re-learned from scratch, whereas the detector already computes this cue explicitly. We reuse it directly.

We retain $\mathbf { p } _ { i }$ at this handoff: it is added to each detection token as it enters the ID decoder, so every track carries its frame position, taken from its frame of origin, into association. Position then travels with the content embedding through the decoder’s attention and feedforward layers, an explicit spatial signal the model would otherwise have to recover from appearance alone.

## 4 Experiments

We first describe the training data and evaluation protocol, then compare with state-of-the-art methods across benchmarks spanning varied tracking regimes (Sec. 4.2), analyze the speedaccuracy trade-off across model sizes (Sec. 4.3), and finally isolate each contribution through an ablation study on PersonPath22 and DanceTrack (Sec. 4.4).

## 4.1 Experimental Setup

Datasets. We evaluate on three benchmarks that stress different facets of association: DanceTrack [45] (similar appearance with complex non-linear motion), SportsMOT [7] (fast non-linear motion in team sports), and PersonPath22 [41] (large-scale, diverse real-world scenes). We train one model per benchmark on its own training split, additionally co-training the PersonPath22 model with CrowdHuman [39]. We report MOTIP2-L as our default variant for state-of-the-art comparisons unless otherwise specified, with MOTIP2-S used for ablations (Sec. 4.4).

Metrics. We primarily report Higher Order Tracking Accuracy (HOTA) [27], which balances detection accuracy (DetA) and association accuracy (AssA); we also report MOTA [2] and IDF1 [36].

Network. We initialize DEIM [13] from its official COCO [20] pre-trained weights. The ID decoder mirrors the DEIM decoder’s hidden dimension and number of layers, so both components scale together across model variants.

Training. We run all experiments on 4 NVIDIA V100 GPUs. Training follows a two-stage schedule: we first pretrain the DEIM detector alone, then jointly train the full model with AdamW [25], gradient clipping, and exponential moving average (EMA). Full optimizer, schedule, clip-sampling, and augmentation settings are detailed in Appendix A.

Hyperparameters. The Spatial ID Loss uses $\lambda _ { \mathrm { C E } } { = } 1 . 0 , \lambda _ { \mathrm { G I o U } } { = } 2 . 0$ , and $\lambda _ { \mathrm { L 1 } } { = } 5 . 0$ , matching the DEIM detection-loss weights. Spatial ID Switches use a confusion radius $\sigma { = } 0 . 5$ , tracklet confirmation requires $N _ { \mathrm { i n i t } } { = } 2$ consecutive frames, greedy ID assignment requires a minimum identity probability $\lambda _ { \mathrm { { I D } } } \mathrm { { = } } 0 . 1$ , and the RoPE timescale is rescaled by $\alpha { = } 2 . 0$ at inference. These, together with the inference resolution (shorter side 800 pixels, longer side capped at 1440, matching both tracking-by-detection [1, 58] and end-to-end [9, 10] protocols), are shared across all benchmarks. Only three settings track the benchmark: the training window and inference miss tolerance scale with typical occlusion length $\scriptstyle ( T = W = 3 0$ frames for DanceTrack and PersonPath22, 60 for SportsMOT); the ID dictionary size scales with crowd density $( K { = } 3 0 0$ for the crowded PersonPath22, 50 otherwise); and the detection and newborn thresholds $( \lambda _ { \mathrm { d e t } } , \lambda _ { \mathrm { n e w b o r n } } )$ are (0.4,0.6) for PersonPath22 and (0.3,0.7) for DanceTrack and SportsMOT.

## 4.2 Comparisons with State-of-the-art Methods

We compare MOTIP2 with state-of-the-art methods on the DanceTrack, SportsMOT, and PersonPath22 test splits, each using the corresponding model described above.

DanceTrack. DanceTrack stresses association under near-identical appearance and highly non-linear motion, the regime where learned association should help most (Tab. 1). Trained on DanceTrack alone, our default MOTIP2-L already reaches 73.4 HOTA, above the strongest prior end-to-end method (FDTA, 71.7) and every tracking-by-detection tracker. The larger MOTIP2-X raises this to 74.8 HOTA, 84.3 DetA, 66.5 AssA, and 78.7 IDF1, the best of any method trained without extra data; its HOTA surpasses even the extra-data $\mathrm { F D T A } ^ { \dag } \ ( 7 4 . { \dot { 4 } } )$ Disabling all three priors while keeping every other component isolates their contribution: MOTIP2-L drops to 71.5 HOTA, below FDTA, and MOTIP2-X to 72.7, losing 3.2 and 3.5 AssA while DetA moves by at most 0.2. The margin over prior end-to-end trackers is therefore associative, not an effect of the DEIM detector. Training additionally on the validation split (the † extra-data protocol) lifts both variants further, with $\mathbf { M O T I P 2 - X } ^ { \dagger }$ setting a new overall state of the art at 77.1 HOTA, 85.4 DetA, 69.8 AssA, and 81.2 IDF1 $( \mathbf { M O T I P 2 - L } ^ { \dagger }$ reaches 76.1 HOTA).

Both variants keep gaining under the extra-data protocol (+2.3 for MOTIP2-X, +2.7 for MOTIP2-L) rather than saturating. Across both regimes the gains are largest in association:

<table><tr><td>Method</td><td>HOTA</td><td>DetA</td><td>AssA</td><td>MOTA</td><td>IDF1</td></tr><tr><td>Tracking-by-detection</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FairMOT []</td><td>39.7</td><td>66.7</td><td>23.8</td><td>82.2</td><td>40.8</td></tr><tr><td>CenterTrack [[]</td><td>41.8</td><td>78.1</td><td>22.6</td><td>86.8</td><td>35.7</td></tr><tr><td>ByteTrack []</td><td>47.7</td><td>71.0</td><td>32.1</td><td>89.6</td><td>53.9</td></tr><tr><td>OC-SORT []</td><td>55.1</td><td>80.3</td><td>38.3</td><td>92.0</td><td>54.6</td></tr><tr><td>StrongSORT []</td><td>55.6</td><td>80.7</td><td>38.6</td><td>91.1</td><td>55.2</td></tr><tr><td>C-BIoU []</td><td>60.6</td><td>81.3</td><td>45.4</td><td>91.6</td><td>61.6</td></tr><tr><td>C-TWiX [[]</td><td>62.1</td><td>81.8</td><td>47.2</td><td>91.4</td><td>63.6</td></tr><tr><td>DiffMOT []</td><td>62.3</td><td>82.5</td><td>47.2</td><td>92.8</td><td>63.0</td></tr><tr><td>UCMCTrack []</td><td>63.6</td><td></td><td>一</td><td>88.9</td><td>65.0</td></tr><tr><td>FastTracker []</td><td>65.9</td><td></td><td>一</td><td>93.4</td><td>67.2</td></tr><tr><td>End-to-end</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MOTR []</td><td>54.2</td><td>73.5</td><td>40.2</td><td>79.7</td><td>51.5</td></tr><tr><td>FastTrackTr [9]</td><td>62.4</td><td>78.4</td><td>49.8</td><td>88.8</td><td>64.8</td></tr><tr><td>CO-MOT [ [四9]</td><td>65.3</td><td>80.1</td><td>53.5</td><td>89.3</td><td>66.5</td></tr><tr><td>MeMOTR []</td><td>68.5</td><td>80.5</td><td>58.4</td><td>89.9</td><td>71.2</td></tr><tr><td>MOTIP [日]</td><td>69.6</td><td>80.4</td><td>60.4</td><td>90.6</td><td>74.7</td></tr><tr><td>FDTA []</td><td>71.7</td><td>81.0</td><td>63.5</td><td>91.3</td><td>77.2</td></tr><tr><td>MOTIP2-L (w/o spatial priors)</td><td>71.5</td><td>83.0</td><td>61.7</td><td>91.6</td><td>73.8</td></tr><tr><td>MOTIP2-L</td><td>73.4</td><td>83.1</td><td>64.9</td><td>91.6</td><td>77.0</td></tr><tr><td>MOTIP2-X (w/o spatial priors)</td><td>72.7</td><td>84.1</td><td>63.0</td><td>92.4</td><td>75.8</td></tr><tr><td>MOTIP2-X</td><td>74.8</td><td>84.3</td><td>66.5</td><td>92.5</td><td>78.7</td></tr><tr><td>End-to-end with extra data</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { C O - M O T ^ { \dagger } }$  [49]</td><td>69.4</td><td>82.1</td><td>58.9</td><td>91.2</td><td>71.9</td></tr><tr><td> $\mathbf { M O T R v } 2 ^ { \dagger }$  [四]</td><td>69.9</td><td>83.0</td><td>59.0</td><td>91.9</td><td>71.7</td></tr><tr><td> $\mathbf { M O T R v } 3 ^ { \dagger }$  [回]</td><td>70.4</td><td>83.8</td><td>59.3</td><td>92.9</td><td>72.3</td></tr><tr><td> $\mathrm { M O T I P ^ { \dagger } \ [ \pmb { \mathbb { U } } ] }$ </td><td>72.0</td><td>81.8</td><td>63.5</td><td>91.9</td><td>76.8</td></tr><tr><td> $\mathrm { F D T A } ^ { \dag } \ [ \pm \mathbf { \ m } ]$ </td><td>74.4</td><td>82.7</td><td>67.0</td><td>92.2</td><td>80.0</td></tr><tr><td> $\scriptstyle \mathbf { M O T I P 2 - L } ^ { \dagger }$ </td><td>76.1</td><td>84.9</td><td>68.4</td><td>92.9</td><td>79.6</td></tr><tr><td> $\mathbf { M O T I P 2 { - } } \mathbf { X } ^ { \dagger }$ </td><td>77.1</td><td>85.4</td><td>69.8</td><td>93.3</td><td>81.2</td></tr></table>

Table 1: State-of-the-art comparison on the DanceTrack [45] test set. The best end-to-end result is in bold. † denotes extra training data (MOTIP2: DanceTrack train+val). MOTIP2- L/X (w/o spatial priors) disable all three priors under the same training setup, keeping every other component, which isolates their contribution from the DEIM detector.

MOTIP2-X lifts AssA by +6.1 over MOTIP and +3.0 over FDTA, while the detection gains track the stronger DEIM backbone.

SportsMOT. SportsMOT couples fast non-linear motion with frequent occlusions in basketball, volleyball, and soccer, stressing association under abrupt direction changes rather than crowd density (Tab. 2). Trained on SportsMOT alone, MOTIP2-L is the best end-to-end tracker on every metric and the best overall on the identity-centric ones: 76.0 HOTA (+1.8 over FDTA, +3.4 over MOTIP, and +3.9 over the strongest tracking-by-detection method, DiffMOT [28]), 85.6 DetA, 67.5 AssA, 93.8 MOTA, and 78.9 IDF1. As on DanceTrack, association accounts for most of the margin (+7.0 AssA over DiffMOT).

<table><tr><td rowspan=1 colspan=1>Method           HOTADetAAssAMOTAIDF1</td></tr><tr><td rowspan=1 colspan=1>Tracking-by-detectionFairMOT []      49.3  70.2 34.7  86.4  53.5</td></tr><tr><td rowspan=1 colspan=1>GTR []           54.5  64.8 45.9  67.9  55.8</td></tr><tr><td rowspan=1 colspan=1>QDTrack []      60.4  77.5 47.2  90.1  62.3</td></tr><tr><td rowspan=1 colspan=1>CenterTrack [[]   62.7  82.1 48.0  90.8  60.0</td></tr><tr><td rowspan=1 colspan=1>ByteTrack []     62.8  77.1 51.2  94.1  69.8</td></tr><tr><td rowspan=1 colspan=1>OC-SORT []      68.1  84.8 54.8  93.4  68.0</td></tr><tr><td rowspan=1 colspan=1>BoT-SORT [0]      68.7  84.4 55.9  94.5  70.0</td></tr><tr><td rowspan=1 colspan=1>DiffMOT [8]      72.1  86.0 60.5  94.5  72.8</td></tr><tr><td rowspan=1 colspan=1>End-to-endTransTrack [4]    68.9  82.7 57.5  92.6  71.5</td></tr><tr><td rowspan=1 colspan=1>SambaMOTR []  69.8  82.2 59.4  90.3  71.9</td></tr><tr><td rowspan=1 colspan=1>MeMOTR []      70.0  83.1 59.1  91.5  71.4</td></tr><tr><td rowspan=1 colspan=1>MOTIP []        72.6  83.5 63.2  92.4  77.1</td></tr><tr><td rowspan=1 colspan=1>FDTA []         74.2  84.1 65.5  93.0  78.5</td></tr><tr><td rowspan=1 colspan=1>MOTIP2-L         76.0  85.6 67.5  93.8  78.9</td></tr></table>

Table 2: State-of-the-art comparison on the SportsMOT [7] test set. The best end-to-end result is in bold.

<table><tr><td>Method IDF1 MOTA IDSW↓ FP↓</td></tr><tr><td>Tracking-by-detection</td></tr><tr><td>CenterTrack [[]] 46.4 59.3 10319 24K 72K</td></tr><tr><td>FairMOT [] 61.1 61.8 5095 15K</td></tr><tr><td>ByteTrack [] 66.8 75.4 5931 17K 40K</td></tr><tr><td>OC-SORT [] 68.2 74.0 4931 17K 41K Path Consistency [26] 69.4 75.8 4663 15K 38K</td></tr><tr><td></td></tr><tr><td>End-to-end TrackFormer [四9] 57.1 69.7 8633 23K47K</td></tr><tr><td>MOTIP2-L 71.1 75.8 5 045 18K 40K</td></tr><tr><td></td></tr></table>

Table 3: State-of-the-art comparison on the PersonPath22 [41] test set. The best end-toend result is in bold. Numbers for comparison methods are compiled from [41] and [26]; these works do not report HOTA-family metrics. For completeness, MOTIP2-L additionally achieves HOTA 62.8, DetA 66.1, and AssA 60.4.

PersonPath22. PersonPath22 is a large-scale real-world benchmark (236 videos spanning diverse indoor and outdoor scenes, lighting and weather conditions, camera viewpoints, and crowd densities) targeting smart-home and smart-city deployment (Tab. 3). As the only modern end-to-end tracker reported on it, MOTIP2-L improves substantially over TrackFormer (IDF1 71.1 vs 57.1, MOTA 75.8 vs 69.7) and attains the best IDF1 overall, +1.7 over the strongest dedicated tracker, Path Consistency [26] (69.4), while matching its MOTA (75.8). The count-based metrics show the expected trade-off of identity-centric association: MOTIP2- L keeps identity switches competitive (5045, fewer than ByteTrack and TrackFormer) but produces more false positives than the most conservative trackers. Prior work here does not report HOTA-family metrics; for reference, MOTIP2-L reaches 62.8 HOTA (66.1 DetA, 60.4 AssA).

<table><tr><td rowspan="2">Method</td><td colspan="3">Accuracy</td><td colspan="3">V100 PT FP32</td><td colspan="3">T4 TRT FP16</td></tr><tr><td>HOTA</td><td>DetA</td><td>AssA</td><td>Det.</td><td>ID Dec</td><td>Total</td><td>Det.</td><td>ID Dec</td><td>Total</td></tr><tr><td>MOTIP [日]</td><td>64.0</td><td>75.4</td><td>54.7</td><td>72.7</td><td>14.2</td><td>86.9</td><td></td><td>一</td><td></td></tr><tr><td>MOTIP2-X</td><td>70.1</td><td>80.7</td><td>61.1</td><td>71.2</td><td>9.9</td><td>81.1</td><td>38.8</td><td>1.3</td><td>40.1</td></tr><tr><td>MOTIP2-L</td><td>68.0</td><td>77.3</td><td>60.0</td><td>45.0</td><td>9.8</td><td>54.8</td><td>21.9</td><td>1.3</td><td>23.2</td></tr><tr><td>MOTIP2-M</td><td>67.3</td><td>76.4</td><td>59.7</td><td>32.5</td><td>6.8</td><td>39.3</td><td>15.8</td><td>0.9</td><td>16.7</td></tr><tr><td>MOTIP2-S</td><td>64.5</td><td>74.0</td><td>56.5</td><td>21.8</td><td>5.3</td><td>27.1</td><td>9.4</td><td>0.7</td><td>10.1</td></tr><tr><td>MOTIP2-N</td><td>59.2</td><td>69.3</td><td>50.9</td><td>14.9</td><td>3.9</td><td>18.8</td><td>4.0</td><td>0.4</td><td>4.4</td></tr></table>

Table 4: Per-component latency breakdown. HOTA on DanceTrack-val and per-component latency (ms). Det. = detector (Deformable DETR for MOTIP, DEIM for MOTIP2); PT = PyTorch, TRT = TensorRT. MOTIP has no standard TensorRT path. Input resolution: $8 0 0 \times 1 4 4 0$ . ID Decoder profiled with T=30 frames, 50 trajectories, and 50 detections.

## 4.3 Speed and Accuracy

Trackers typically run online, on a live video stream feeding downstream tasks, and must keep pace with the incoming frames; we therefore evaluate accuracy jointly with latency.

Figure 1 plots the HOTA-versus-latency trade-off on DanceTrack-val (the state-of-the-art tables above report the test split), with the exact per-model HOTA and latency behind each point listed in Appendix F. Across model sizes, MOTIP2 traces a Pareto front ahead of prior end-to-end trackers (FastTrackTr [19], PuTR [22]): MOTIP2-S matches MOTIP’s accuracy while running over 3× faster (27.1 vs 86.9 ms), and our larger variants surpass them in accuracy while remaining faster. The association advantage holds even before detection catches up: with a shallower ID decoder than MOTIP (3 vs 6 layers), MOTIP2-S surpasses MOTIP’s HOTA despite 1.4 lower DetA (+1.8 AssA), and once detection reaches MOTIP’s level (MOTIP2-M) association leads by +5.0 AssA. The improvement is concentrated in association, not the DEIM backbone.

Latency breakdown. Table 4 reports per-component latency for MOTIP2 across V100 (PyTorch FP32) and T4 (TensorRT FP16), profiling DEIM and the ID decoder separately; a full comparison against baselines is given in Appendix F. DEIM latency is fixed once the model size is chosen, while the ID decoder’s latency grows roughly linearly with the number of tracked objects in the trajectory history. Sparser scenes such as DanceTrack therefore run faster, while denser scenes incur only a moderate increase. MOTIP has no entry in the TensorRT columns: its Deformable DETR detector relies on deformable attention, which has no standard TensorRT path, so it cannot be exported or benchmarked on a deployment GPU. Both MOTIP2 components export to TensorRT as separate engines, where MOTIP2-S reaches 10.1 ms (≈ 99 FPS) on a T4 under FP16.

## 4.4 Ablation Studies

All ablations use MOTIP2-S. We train one model per evaluation benchmark following its standard recipe: one on PersonPath22+CrowdHuman, scored on the PersonPath22 test set, and one on the DanceTrack training split, scored on the DanceTrack validation set. Every ablated variant is retrained under both recipes and evaluated on its corresponding split.

The three spatial priors all inject geometric structure into association (Tab. 5). Removing all three at once costs −2.9 HOTA and −5.8 AssA on PersonPath22 and −1.5 HOTA and −2.5 AssA on DanceTrack; Appendix D shows the qualitative effect on a DanceTrack sequence, where the no-prior baseline teleports an identity across the frame. The loss- and representation-stage priors hold up even with the data-stage prior neutralized: against the matched reference (w/o IDSW on PersonPath22, uniform IDSW on DanceTrack, both of which keep the loss and anchor under the same data-stage setting), dropping them costs a further −0.8 / −0.7 HOTA and −1.3 / −1.8 AssA, so their benefit is not merely an artifact of the spatial augmentation. The two benchmarks expose them differently: PersonPath22’s wide, crowded scenes make the data-stage prior decisive, while DanceTrack’s compact crowds of near-identical dancers bring out the loss- and representation-stage priors. We discuss each in turn, in the order of Sec. 3, then ablate two architectural tweaks to the ID decoder and inference separately (Sec. 4.5).

<table><tr><td></td><td colspan="4">PersonPath22</td><td colspan="4">DanceTrack</td></tr><tr><td></td><td>HOTA</td><td>DetA</td><td>AssA</td><td>IDF1</td><td>HOTA</td><td>DetA</td><td>AssA</td><td>IDF1</td></tr><tr><td>Full model</td><td>61.6</td><td>63.8</td><td>60.3</td><td>70.4</td><td>64.5</td><td>74.0</td><td>56.5</td><td>68.8</td></tr><tr><td>uniform IDSW</td><td>56.1</td><td>58.6</td><td>54.4</td><td>64.0</td><td>63.7</td><td>73.0</td><td>55.8</td><td>68.1</td></tr><tr><td>w/o IDSW</td><td>59.5</td><td>64.2</td><td>55.8</td><td>66.8</td><td>60.1</td><td>73.3</td><td>49.7</td><td>63.0</td></tr><tr><td>w/o Spatial ID Loss</td><td>61.4</td><td>64.2</td><td>59.4</td><td>70.1</td><td>63.5</td><td>73.3</td><td>55.5</td><td>67.3</td></tr><tr><td>w/o Spatial Anchor</td><td>61.6</td><td>64.3</td><td>59.9</td><td>70.4</td><td>64.0</td><td>73.3</td><td>56.1</td><td>68.6</td></tr><tr><td>w/o all priors†</td><td>58.7</td><td>64.1</td><td>54.5</td><td>65.7</td><td>63.0</td><td>74.0</td><td>54.0</td><td>66.9</td></tr></table>

Table 5: Ablation of the spatial priors. MOTIP2-S, each variant trained and evaluated per benchmark (PersonPath22+CrowdHuman; DanceTrack train split). uniform IDSW downgrades our Spatial ID Switches to a non-spatial permutation; w/o IDSW removes the augmentation entirely. <sup>†</sup>With all priors off, the data-stage prior is removed on PersonPath22 but set to uniform on DanceTrack, where uniform is near-neutral rather than harmful (cf. rows above). The full model (bold row label) is the reference each removal is compared against.

Spatial ID Switches. Spatial ID Switches are by far the most important prior on PersonPath22, where they also expose a failure mode of the uniform permutation they replace. Removing the augmentation altogether costs −2.1 HOTA (almost all association, −4.5 AssA), but reverting it to MOTIP’s uniform permutation is worse still: at 56.1 HOTA it falls 3.4 points below the no-augmentation baseline (59.5) and drags detection down with it (−5.6 DetA). On PersonPath22 a uniform permutation is therefore harmful, not just uninformative: it destabilizes joint training instead of teaching the decoder to recover from confusions (training curves in Appendix C). Figure 4 shows the three behaviors on a representative crossing: only our model recovers the correct identities, the no-augmentation variant trusts its trajectory history blindly and never undoes the swap, and the uniform variant grows uncertain through the occlusion, losing the detection before switching identity.

The cause is the train/inference mismatch that GIoU sampling removes (Sec. 3.2): Person-Path22 spreads objects across the frame, so uniform permutations fabricate mostly implausible long-distance swaps. Restricting confusions to spatially plausible pairs recovers +5.5 HOTA over the uniform variant. Training on the confusions the tracker meets at inference concentrates the decoder’s uncertainty on the few tracks an object actually crosses, rather than on every identity in the frame. DanceTrack shows why the effect is dataset-dependent: the augmentation is again critical (−4.4 HOTA when removed), but its uniform variant costs only −0.8, because dancers stay clustered, so even a random permutation swaps mostly nearby objects.

![](images/07f5fb0b8e6190fb6a0b45e145adb7bf319317b0f29ba902cc4ef8d528869466.jpg)  
Figure 4: Qualitative tracking comparison on PersonPath22 (uid\_vid\_00063). Rows (top to bottom): our GIoU-based sampling (Ours), no augmentation (w/o IDSW), and MOTIP’s uniform permutation (uniform IDSW). Columns are ordered in time, left to right. Two pedestrians cross and occlude; only ours recovers the correct identities afterward.

Spatial ID Loss. The Spatial ID Loss (Sec. 3.3) mainly improves association and complements the dominant Spatial ID Switches: removing it costs only −0.2 HOTA and −0.9 AssA on PersonPath22, but −1.0 HOTA and −1.0 AssA on DanceTrack, where near-identical appearance makes geometric supervision more valuable.

Spatial Anchor. Reinjecting the detector’s per-frame positional encoding into every IDdecoder token is the lightest of the three priors: no new parameters, no new supervision, just reusing an encoding the detector has already produced. We expect a correspondingly modest gain, concentrated where many candidate tracks share similar content embeddings and an explicit spatial cue is most discriminative. The data agree: removal costs −0.4 AssA and no HOTA on PersonPath22, but −0.5 HOTA and −0.4 AssA on DanceTrack, where appearance alone is weakly discriminative.

## 4.5 Decoder and Inference Ablations

Two changes to MOTIP2’s ID decoder and inference fall outside the spatial priors: rotary temporal encoding in the decoder and tracklet confirmation at inference.

Rotary Position Embedding. Replacing MOTIP’s learned temporal bias with RoPE lifts AssA by +1.0 and HOTA by +0.3 at the best timescale (Tab. 6). The gain hinges on the inference timescale α: training samples wider temporal intervals than inference, so the raw encoding $( \alpha { = } 1 . 0 )$ trails the learned bias, while rescaling toward the training mean recovers and surpasses it, plateauing across $\alpha \in [ 2 . 0 , 2 . 5 ]$ . We adopt $\alpha { = } 2 . 0$

<table><tr><td>Method</td><td>α</td><td>HOTA</td><td>DetA</td><td>AssA</td><td>MOTA</td><td>IDF1</td></tr><tr><td>Rel. PE</td><td>一</td><td>61.3</td><td>64.1</td><td>59.3</td><td>73.4</td><td>70.2</td></tr><tr><td rowspan="4">RoPE</td><td>1.0</td><td>60.4</td><td>63.7</td><td>58.0</td><td>72.7</td><td>68.7</td></tr><tr><td>1.5</td><td>61.0</td><td>63.7</td><td>59.1</td><td>72.8</td><td>69.6</td></tr><tr><td>2.0</td><td>61.6</td><td>63.8</td><td>60.3</td><td>72.9</td><td>70.4</td></tr><tr><td>2.5</td><td>61.6</td><td>63.8</td><td>60.3</td><td>72.9</td><td>70.4</td></tr></table>

Table 6: Effect of positional encoding type and time rescaling factor α. Results on PersonPath22 test. We compare RoPE with the learned relative positional encoding from MOTIP [10] (Rel. PE). The best result for each metric is shown in bold.

Tracklet Confirmation. Requiring $N _ { \mathrm { i n i t } }$ consecutive frames of consistent assignment before committing a tentative track trims identity switches without hurting accuracy: from $N _ { \mathrm { i n i t } } { = } 1$ to 3, identity switches drop by 12.5% on PersonPath22 and 28.3% on DanceTrack, while HOTA stays flat and IDF1 and MOTA edge up only slightly. Evicting unconfirmed tracks also keeps the historical context clean, freeing ID-bank slots that short-lived false detections would otherwise occupy. Accuracy plateaus at $N _ { \mathrm { i n i t } } { = } 2$ , and identity switches fall only marginally beyond it, so we adopt $N _ { \mathrm { i n i t } } { = } 2$ ; the full sweep is in Appendix E.

## 5 Conclusion

We argue that the association errors separating end-to-end trackers from tracking-by-detection are mainly geometric, and we correct them with three explicit spatial priors. Which prior helps most depends on scene geometry rather than the benchmark. In wide, crowded scenes the data-stage prior is decisive: the uniform shuffling it replaces trains the model on implausible long-distance swaps and makes training unstable, whereas restricting confusions to nearby boxes recovers +5.5 HOTA on PersonPath22. When objects look nearly identical, the other two matter more. The loss-stage prior makes a confident long-distance error cost more than a nearby one, and the representation-stage prior ties each track to its image position; both separate identities that appearance alone confuses. This reliance on proximity also sets the failure case we expect to be hardest: a simultaneous exit and entry at the same location, where proximity draws the leaving identity onto the new object. The result is a family of end-to-end trackers spanning the speed-accuracy trade-off, from a real-time model to variants that set a new state of the art on identity association across all three benchmarks.

## Acknowledgements

This work was granted access to the HPC resources of IDRIS under the allocations AD011014610 and A0191013894 made by GENCI.

## References

[1] Nir Aharon, Roy Orfaig, and Ben-Zion Bobrovsky. Bot-sort: Robust associations multipedestrian tracking. CoRR, abs/2206.14651, 2022. doi: 10.48550/ARXIV.2206.14651. URL https://doi.org/10.48550/arXiv.2206.14651.

[2] Keni Bernardin and Rainer Stiefelhagen. Evaluating multiple object tracking performance: The CLEAR MOT metrics. EURASIP J. Image Video Process., 2008, 2008. doi: 10.1155/2008/246309. URL https://doi.org/10.1155/2008/246309.

[3] Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft. Simple online and realtime tracking. In 2016 IEEE International Conference on Image Processing, ICIP 2016, Phoenix, AZ, USA, September 25-28, 2016, pages 3464–3468. IEEE, 2016. doi: 10.1109/ICIP.2016.7533003. URL https://doi.org/10.1109/ ICIP.2016.7533003.

[4] Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, and Kris Kitani. Observation-centric SORT: rethinking SORT for robust multi-object tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 9686–9696. IEEE, 2023. doi: 10.1109/CVPR52729.2023.00934. URL https://doi.org/10.1109/ CVPR52729.2023.00934.

[5] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23- 28, 2020, Proceedings, Part I, volume 12346 of Lecture Notes in Computer Science, pages 213–229. Springer, 2020. doi: 10.1007/978-3-030-58452-8\_13. URL https://doi.org/10.1007/978-3-030-58452-8\_13.

[6] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. CoRR, abs/2306.15595, 2023. doi: 10.48550/ARXIV.2306.15595. URL https://doi. org/10.48550/arXiv.2306.15595.

[7] Yutao Cui, Chenkai Zeng, Xiaoyu Zhao, Yichun Yang, Gangshan Wu, and Limin Wang. Sportsmot: A large multi-object tracking dataset in multiple sports scenes. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 9887–9897. IEEE, 2023. doi: 10.1109/ICCV51070.2023.00910. URL https://doi.org/10.1109/ICCV51070.2023.00910.

[8] Yunhao Du, Zhicheng Zhao, Yang Song, Yanyun Zhao, Fei Su, Tao Gong, and Hongying Meng. Strongsort: Make deepsort great again. IEEE Trans. Multim., 25:8725–8737, 2023. doi: 10.1109/TMM.2023.3240881. URL https://doi.org/10.1109/ TMM.2023.3240881.

[9] Ruopeng Gao and Limin Wang. Memotr: Long-term memory-augmented transformer for multi-object tracking. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 9867–9876. IEEE, 2023. doi: 10.

1109/ICCV51070.2023.00908. URL https://doi.org/10.1109/ICCV51070. 2023.00908.

[10] Ruopeng Gao, Ji Qi, and Limin Wang. Multiple object tracking as ID prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 27883–27893. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.02596. URL https:// openaccess.thecvf.com/content/CVPR2025/html/Gao\_Multiple\_ Object\_Tracking\_as\_ID\_Prediction\_CVPR\_2025\_paper.html.

[11] Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. YOLOX: exceeding YOLO series in 2021. CoRR, abs/2107.08430, 2021. URL https://arxiv.org/ abs/2107.08430.

[12] Hamidreza Hashempoor and Yu Dong Hwang. FastTracker: Real-time and accurate visual tracking. CoRR, abs/2508.14370, 2025. URL https://arxiv.org/abs/ 2508.14370.

[13] Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, and Xi Shen. DEIM: DETR with improved matching for fast convergence. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 15162–15171. Computer Vision Foundation / IEEE, 2025. doi: 10.1109/CVPR52734.2025.01412. URL https://openaccess.thecvf. com/content/CVPR2025/html/Huang\_DEIM\_DETR\_with\_Improved\_ Matching\_for\_Fast\_Convergence\_CVPR\_2025\_paper.html.

[14] Junjie Jiang, Zelin Wang, Manqi Zhao, Yin Li, and Dongsheng Jiang. SAM2MOT: A novel paradigm of multi-object tracking by segmentation. In Proceedings of the Fortieth AAAI Conference on Artificial Intelligence, AAAI 2026, volume 40, pages 5388– 5396. AAAI Press, 2026. URL https://ojs.aaai.org/index.php/AAAI/ article/view/37455.

[15] Rudolf E. Kalman. A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1):35–45, 1960. doi: 10.1115/1.3662552. URL https: //doi.org/10.1115/1.3662552.

[16] Harold W. Kuhn. The hungarian method for the assignment problem. In Michael Jünger, Thomas M. Liebling, Denis Naddef, George L. Nemhauser, William R. Pulleyblank, Gerhard Reinelt, Giovanni Rinaldi, and Laurence A. Wolsey, editors, 50 Years ofInteger Programming 1958-2008 - From the Early Years to the State-of-the-Art, pages 29–47. Springer, 2010. doi: 10.1007/978-3-540-68279-0\_2. URL https://doi.org/10. 1007/978-3-540-68279-0\_2.

[17] Feng Li, Hao Zhang, Shilong Liu, Jian Guo, Lionel M. Ni, and Lei Zhang. DN-DETR: accelerate DETR training by introducing query denoising. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 13609–13617. IEEE, 2022. doi: 10.1109/CVPR52688.2022.01325. URL https://doi.org/10.1109/CVPR52688.2022.01325.

[18] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified

and distributed bounding boxes for dense object detection. In NeurIPS, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ f0bda020d2470f2e74990a07a607ebd9-Abstract.html.

[19] Pan Liao, Feng Yang, Di Wu, Jinwen Yu, Xingxin Li, and Dingwen Zhang. Fasttracktr: Real-time multiobject tracking with transformers for real world. IEEE Trans. Ind. Informatics, 22(3):1817–1827, 2026. doi: 10.1109/TII.2025.3631698. URL https: //doi.org/10.1109/TII.2025.3631698.

[20] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In David J. Fleet, Tomás Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V, volume 8693 of Lecture Notes in Computer Science, pages 740–755. Springer, 2014. doi: 10.1007/978-3-319-10602-1\_48. URL https://doi.org/10.1007/978-3-319-10602-1\_48.

[21] Tsung-Yi Lin, Priya Goyal, Ross B. Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, pages 2999–3007. IEEE Computer Society, 2017. doi: 10.1109/ICCV.2017.324. URL https://doi.org/10.1109/ICCV. 2017.324.

[22] Chongwei Liu, Haojie Li, Zhihui Wang, and Rui Xu. Is a pure transformer effective for separated and online multi-object tracking? ACM Trans. Multim. Comput. Commun. Appl., 21(12):1–25, 2025. doi: 10.1145/3749105. URL https://doi.org/10. 1145/3749105.

[23] Shilong Liu, Feng Li, Hao Zhang, Xiao Yang, Xianbiao Qi, Hang Su, Jun Zhu, and Lei Zhang. DAB-DETR: dynamic anchor boxes are better queries for DETR. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id=oMI9PjOb9Jl.

[24] Zelin Liu, Xinggang Wang, Cheng Wang, Wenyu Liu, and Xiang Bai. Sparsetrack: Multi-object tracking by performing scene decomposition based on pseudo-depth. IEEE TCSVT, 35(5):4870–4882, 2025. doi: 10.1109/TCSVT.2024.3524670.

[25] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Int. Conf. Learn. Represent., 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

[26] Zijia Lu, Bing Shuai, Yanbei Chen, Zhenlin Xu, and Davide Modolo. Self-supervised multi-object tracking with path consistency. In CVPR, pages 19016–19026, 2024. doi: 10.1109/CVPR52733.2024.01799. URL https://doi.org/10.1109/ CVPR52733.2024.01799.

[27] Jonathon Luiten, Aljosa Osep, Patrick Dendorfer, Philip H. S. Torr, Andreas Geiger, Laura Leal-Taixé, and Bastian Leibe. HOTA: A higher order metric for evaluating multi-object tracking. Int. J. Comput. Vis., 129(2):548–578, 2021. doi: 10.1007/S11263-020-01375-2. URL https://doi.org/10.1007/ s11263-020-01375-2.

[28] Weiyi Lv, Yuhang Huang, Ning Zhang, Ruei-Sung Lin, Mei Han, and Dan Zeng. DiffMOT: A real-time diffusion-based multiple object tracker with non-linear prediction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 19321–19330. IEEE, 2024. doi: 10.1109/CVPR52733.2024.01828. URL https://doi.org/10.1109/ CVPR52733.2024.01828.

[29] Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixé, and Christoph Feichtenhofer. Trackformer: Multi-object tracking with transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 8834–8844. IEEE, 2022. doi: 10.1109/CVPR52688.2022.00864. URL https://doi.org/10.1109/CVPR52688.2022.00864.

[30] Mehdi Miah, Guillaume-Alexandre Bilodeau, and Nicolas Saunier. Learning data association for multi-object tracking using only coordinates. Pattern Recognition, 160: 111169, 2025. doi: 10.1016/j.patcog.2024.111169. URL https://doi.org/10. 1016/j.patcog.2024.111169.

[31] Anton Milan, Laura Leal-Taixé, Ian D. Reid, Stefan Roth, and Konrad Schindler. MOT16: A benchmark for multi-object tracking. CoRR, abs/1603.00831, 2016. URL http://arxiv.org/abs/1603.00831.

[32] Jiangmiao Pang, Linlu Qiu, Xia Li, Haofeng Chen, Qi Li, Trevor Darrell, and Fisher Yu. Quasi-dense similarity learning for multiple object tracking. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 164–173. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.00023. URL https://openaccess.thecvf.com/content/CVPR2021/html/ Pang\_Quasi-Dense\_Similarity\_Learning\_for\_Multiple\_Object\_ Tracking\_CVPR\_2021\_paper.html.

[33] Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, and Feng Wu. D-FINE: redefine regression task of detrs as fine-grained distribution refinement. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum? id=MFZjrTFE7h.

[34] Zheng Qin, Sanping Zhou, Le Wang, Jinghai Duan, Gang Hua, and Wei Tang. Motiontrack: Learning robust short-term and long-term motions for multi-object tracking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 17939–17948. IEEE, 2023. doi: 10.1109/CVPR52729.2023.01720. URL https://doi.org/10.1109/ CVPR52729.2023.01720.

[35] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian D. Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 658–666. Computer Vision Foundation / IEEE, 2019. doi: 10.1109/CVPR.2019.00075. URL http://openaccess.thecvf.com/content\_CVPR\_2019/html/

Rezatofighi\_Generalized\_Intersection\_Over\_Union\_A\_Metric\_ and\_a\_Loss\_for\_CVPR\_2019\_paper.html.

[36] Ergys Ristani, Francesco Solera, Roger S. Zou, Rita Cucchiara, and Carlo Tomasi. Performance measures and a data set for multi-target, multi-camera tracking. In Gang Hua and Hervé Jégou, editors, Computer Vision - ECCV 2016 Workshops - Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part II, volume 9914 of Lecture Notes in Computer Science, pages 17–35, 2016. doi: 10.1007/978-3-319-48881-3\ \_2. URL https://doi.org/10.1007/978-3-319-48881-3\_2.

[37] Mattia Segù, Luigi Piccinelli, Siyuan Li, Yung-Hsu Yang, Luc Van Gool, and Bernt Schiele. Samba: Synchronized set-of-sequences modeling for multiple object tracking. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview. net/forum?id=OeBY9XqiTz.

[38] Jenny Seidenschwarz, Guillem Brasó, Victor Castro Serrano, Ismail Elezi, and Laura Leal-Taixé. Simple cues lead to a strong multi-object tracker. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 13813–13823. IEEE, 2023. doi: 10.1109/CVPR52729.2023.01327. URL https://doi.org/10.1109/CVPR52729.2023.01327.

[39] Shuai Shao, Zijian Zhao, Boxun Li, Tete Xiao, Gang Yu, Xiangyu Zhang, and Jian Sun. Crowdhuman: A benchmark for detecting human in a crowd. CoRR, abs/1805.00123, 2018. URL http://arxiv.org/abs/1805.00123.

[40] Yuqing Shao, Yuchen Yang, Rui Yu, Weilong Li, Xu Guo, Huaicheng Yan, Wei Wang, and Xiao Sun. From detection to association: Learning discriminative object embeddings for multi-object tracking. CoRR, abs/2512.02392, 2025. URL https://arxiv. org/abs/2512.02392.

[41] Bing Shuai, Alessandro Bergamo, Uta Büchler, Andrew G. Berneshawi, Alyssa Boden, and Joseph Tighe. Large scale real-world multi-person tracking. In Shai Avidan, Gabriel J. Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part VIII, volume 13668 of Lecture Notes in Computer Science, pages 504–521. Springer, 2022. doi: 10.1007/978-3-031-20074-8\_29. URL https://doi.org/10.1007/978-3-031-20074-8\_29.

[42] Vukašin Stanojevic and Branimir Todorovi´ c. BoostTrack++: using tracklet information´ to detect more objects in multiple object tracking. Filomat, 39(16):5685–5702, 2025. doi: 10.2298/FIL2516685S. URL https://doi.org/10.2298/FIL2516685S.

[43] Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/J.NEUCOM.2023.127063. URL https://doi. org/10.1016/j.neucom.2023.127063.

[44] Peize Sun, Yi Jiang, Rufeng Zhang, Enze Xie, Jinkun Cao, Xinting Hu, Tao Kong, Zehuan Yuan, Changhu Wang, and Ping Luo. Transtrack: Multiple-object tracking with transformer. CoRR, abs/2012.15460, 2020. URL https://arxiv.org/abs/ 2012.15460.

[45] Peize Sun, Jinkun Cao, Yi Jiang, Zehuan Yuan, Song Bai, Kris Kitani, and Ping Luo. Dancetrack: Multi-object tracking in uniform appearance and diverse motion. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 20961–20970. IEEE, 2022. doi: 10.1109/CVPR52688.2022.02032. URL https://doi.org/10.1109/ CVPR52688.2022.02032.

[46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008, 2017. URL https://proceedings.neurips.cc/paper/2017/ hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[47] Zhongdao Wang, Liang Zheng, Yixuan Liu, Yali Li, and Shengjin Wang. Towards real-time multi-object tracking. In Andrea Vedaldi, Horst Bischof, Thomas Brox, and Jan-Michael Frahm, editors, Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XI, volume 12356 of Lecture Notes in Computer Science, pages 107–122. Springer, 2020. doi: 10.1007/978-3-030-58621-8\ \_7. URL https://doi.org/10.1007/978-3-030-58621-8\_7.

[48] Nicolai Wojke, Alex Bewley, and Dietrich Paulus. Simple online and realtime tracking with a deep association metric. In 2017 IEEE International Conference on Image Processing, ICIP 2017, Beijing, China, September 17-20, 2017, pages 3645–3649. IEEE, 2017. doi: 10.1109/ICIP.2017.8296962. URL https://doi.org/10.1109/ ICIP.2017.8296962.

[49] Feng Yan, Weixin Luo, Yujie Zhong, Yiyang Gan, and Lin Ma. CO-MOT: boosting end-to-end transformer-based multi-object tracking via coopetition label assignment and shadow sets. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https: //openreview.net/forum?id=0ov0dMQ3mN.

[50] Mingzhan Yang, Guangxin Han, Bin Yan, Wenhua Zhang, Jinqing Qi, Huchuan Lu, and Dong Wang. Hybrid-sort: Weak cues matter for online multi-object tracking. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan, editors, Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications ofArtificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 6504–6512. AAAI Press, 2024. doi: 10.1609/AAAI.V38I7. 28471. URL https://doi.org/10.1609/aaai.v38i7.28471.

[51] Kefu Yi, Kai Luo, Xiaolei Luo, Jiangui Huang, Hao Wu, Rongdong Hu, and Wei Hao. UCMCTrack: Multi-object tracking with uniform camera motion compensation. In AAAI, volume 38, pages 6702–6710, 2024. doi: 10.1609/aaai.v38i7.28493.

[52] En Yu, Tiancai Wang, Zhuoling Li, Yuang Zhang, Xiangyu Zhang, and Wenbing Tao. Motrv3: Release-fetch supervision for end-to-end multi-object tracking. CoRR,

abs/2305.14298, 2023. doi: 10.48550/ARXIV.2305.14298. URL https://doi. org/10.48550/arXiv.2305.14298.

[53] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. BDD100K: A diverse driving dataset for heterogeneous multitask learning. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 2633–2642. Computer Vision Foundation / IEEE, 2020. doi: 10.1109/CVPR42600.2020.00271. URL https://openaccess.thecvf.com/content\_CVPR\_2020/html/ Yu\_BDD100K\_A\_Diverse\_Driving\_Dataset\_for\_Heterogeneous\_ Multitask\_Learning\_CVPR\_2020\_paper.html.

[54] Fangao Zeng, Bin Dong, Yuang Zhang, Tiancai Wang, Xiangyu Zhang, and Yichen Wei. MOTR: end-to-end multiple-object tracking with transformer. In Shai Avidan, Gabriel J. Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXVII, volume 13687 of Lecture Notes in Computer Science, pages 659–675. Springer, 2022. doi: 10.1007/978-3-031-19812-0\_38. URL https://doi.org/10.1007/978-3-031-19812-0\_38.

[55] Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. DINO: DETR with improved denoising anchor boxes for endto-end object detection. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=3mRwyG5one.

[56] Haoyang Zhang, Ying Wang, Feras Dayoub, and Niko Sünderhauf. Varifocal-Net: An IoU-aware dense object detector. In CVPR, pages 8514–8523, 2021. URL https://openaccess.thecvf.com/content/CVPR2021/html/ Zhang\_VarifocalNet\_An\_IoU-Aware\_Dense\_Object\_Detector\_ CVPR\_2021\_paper.html.

[57] Yifu Zhang, Chunyu Wang, Xinggang Wang, Wenjun Zeng, and Wenyu Liu. Fairmot: On the fairness of detection and re-identification in multiple object tracking. Int. J. Comput. Vis., 129(11):3069–3087, 2021. doi: 10.1007/S11263-021-01513-4. URL https://doi.org/10.1007/s11263-021-01513-4.

[58] Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, and Xinggang Wang. Bytetrack: Multi-object tracking by associating every detection box. In Shai Avidan, Gabriel J. Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal Hassner, editors, Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part XXII, volume 13682 of Lecture Notes in Computer Science, pages 1–21. Springer, 2022. doi: 10.1007/978-3-031-20047-2\_1. URL https://doi.org/10.1007/ 978-3-031-20047-2\_1.

[59] Yuang Zhang, Tiancai Wang, and Xiangyu Zhang. Motrv2: Bootstrapping end-to-end multi-object tracking by pretrained object detectors. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 22056–22065. IEEE, 2023. doi: 10.1109/CVPR52729.2023.02112. URL https://doi.org/10.1109/CVPR52729.2023.02112.

[60] Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. Detrs beat yolos on real-time object detection. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 16965–16974. IEEE, 2024. doi: 10.1109/CVPR52733.2024. 01605. URL https://doi.org/10.1109/CVPR52733.2024.01605.

[61] Xingyi Zhou, Vladlen Koltun, and Philipp Krähenbühl. Tracking objects as points. In Computer Vision - ECCV 2020 - 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part IV, volume 12349 of Lecture Notes in Computer Science, pages 474–490. Springer, 2020. doi: 10.1007/978-3-030-58548-8\_28. URL https: //doi.org/10.1007/978-3-030-58548-8\_28.

[62] Xingyi Zhou, Tianwei Yin, Vladlen Koltun, and Philipp Krähenbühl. Global tracking transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 8761–8770. IEEE, 2022. doi: 10.1109/CVPR52688.2022.00857. URL https://doi.org/10.1109/ CVPR52688.2022.00857.

[63] Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, and Jifeng Dai. Deformable DETR: deformable transformers for end-to-end object detection. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id= gZ9hCDWe6ke.

# MOTIP2: Spatial Priors for End-to-End Multi-Object Tracking

# Supplementary Material

## A Training Details

Optimizer Configuration. We use AdamW [25] with per-parameter-group learning rates and weight decay. The encoder and decoder use a base learning rate of $8 \times 1 0 ^ { - 4 }$ for MOTIP2- N, $2 \times 1 0 ^ { - 4 }$ for MOTIP2-S/M, and $2 . 5 \times 1 0 ^ { - 4 }$ for MOTIP2-L/X. The ID decoder uses twice this rate. The backbone uses 0.5× this rate for MOTIP2-N/S, 0.1× for MOTIP2-M, 0.05× for MOTIP2-L, and 0.01× for MOTIP2-X. Weight decay is set per model size as well and is disabled for normalization layers, bias terms, and embedding parameters. Gradients are clipped to a max norm of 0.1. We apply EMA with a decay of 0.9995, with no warmup during pretraining and a 1000-step warmup during joint training.

Learning Rate Schedule. Training follows a two-stage schedule, both stages at a constant learning rate with 40k samples per epoch. We first pretrain the DEIM [13] detector alone, with no learning-rate warmup, then jointly train the full model (detector + ID decoder) with a 500-iteration linear warmup.

Batch Size. For MOTIP2-S/M/L/X, we use 8 images per GPU during detector pretraining and 2 clips per GPU during joint training, giving global batch sizes of 32 images and 8 clips across four GPUs. MOTIP2-N uses 32 images and 4 clips per GPU, giving global batch sizes of 128 images and 16 clips, respectively. We do not accumulate gradients.

Clip Sampling. We sample clips of 30 frames (60 for SportsMOT) with a temporal interval uniformly sampled from 1 to 4. The detection loss is computed on 4 randomly sampled frames per clip (8 for the nano model), while the ID decoder receives the full clip.

Data Augmentation Details. We apply augmentations at two levels: image-level and trajectory-level.

Image-level geometric. Horizontal flip with probability 0.5; random zoom-out with probability 0.5 and scale range [1.0, 2.0]; IoU-aware random crop with probability 0.8, scale range [0.3, 1.0], and aspect ratio range [1.5, 2.0].

Multi-scale resizing. Every clip is randomly resized to one of a set of resolutions spanning roughly 75% to 125% of the base size, each snapped to a multiple of 32, with a tunable sampling bias toward the base resolution; all frames within a clip share the sampled scale. Exposing the model to many resolutions during training yields a single network that handles a range of input sizes at inference without retraining, so the test-time resolution becomes a free accuracy-latency dial.

Photometric. Applied with probability 0.5: brightness jitter in [0.875,1.125], contrast in [0.5, 1.5], saturation in [0.5, 1.5], and hue shift in [−0.05, 0.05].

Trajectory-level. Spatial ID Switches are applied with probability 1.0, sampling identity permutations weighted by pairwise GIoU affinity. Random occlusion masking is applied with probability 0.5. Trajectories are organized into 6 groups of 100 IDs each.

Loss Weights. The DEIM detection losses use the following weights: classification (MAL) 1.0, bounding box L1 5.0, GIoU 2.0, FGL 0.15, and DDF 1.5 (the last two from D-FINE [33]), with focal loss [21] parameters α=0.75 and γ=1.5. The Hungarian matcher costs are: class 2.0, bounding box 5.0, GIoU 2.0. The ID association loss weight is 1.0. Following qualityaware detection losses [18, 56], the ID cross-entropy is weighted per-sample by the predictedto-ground-truth IoU, concentrating association learning on well-localized detections.

## B Reference Comparison on MOT17

We train MOTIP2-L/X on MOT17 [31] and CrowdHuman [39], matching MOTIP’s recipe, and evaluate on the MOT17 test set (Tab. 7). MOTIP2 improves over MOTIP [10] on every metric, including association (+1.0 AssA and +1.4 IDF1 at L), though the gains concentrate in detection (+1.9 DetA at X). Scaling L to X leaves AssA flat at 58.0: association on MOT17 is data-bound rather than capacity-bound, since CrowdHuman’s static images supervise detection only, leaving MOT17’s 7 training sequences as the sole association supervision, whose near-linear motion favors training-free Kalman [15] association. This regime motivated our focus on PersonPath22, a pedestrian benchmark with 138 training videos.

## C Training Stability of Spatial ID Switches

Figure 5 tracks PersonPath22 HOTA across the joint-training epochs for the three augmentation variants ablated in Sec. 4.4. The uniform permutation of MOTIP [10] improves for the first few epochs, then diverges and ends well below the no-augmentation baseline. Because PersonPath22’s wide scenes spread objects across the frame, a uniform permutation fabricates mostly long-distance swaps that rarely occur at inference (Sec. 3.2); supervised against these implausible histories, the ID decoder destabilizes the jointly trained detector rather than learning to recover from realistic confusions. GIoU-based sampling (ours) restricts confusions to spatially plausible pairs, trains stably throughout, and reaches the best HOTA.

Table 8 quantifies the augmentation over one epoch. The uniform permutation mostly swaps non-overlapping boxes (GIoU ≤ 0), fabricating long-range swaps; Spatial ID Switches realize fewer swaps, focused on overlapping objects (mean GIoU 0.20/0.15), yet reach a higher HOTA. At inference, the uniform variant produces the most long-range ID switches (2.2×/1.7× the no-augmentation counts), whereas ours stays close to the no-augmentation level while learning to recover from identity errors.

## D A Long-Distance Identity Switch

Figure 6 illustrates the long-distance switch raised in Sec. 1, the failure our spatial priors are designed to suppress. On a crowded DanceTrack sequence, the baseline trained without any of the spatial priors teleports an identity nearly across the full frame width between two consecutive frames, with no occlusion and no plausible motion path, merging two distant but near-identical dancers into a single track. Among the three priors, the Spatial ID Loss most directly targets this error by restoring the geometric cost that cross-entropy leaves out (Sec. 3.3). With all three priors active, our full model keeps both identities stable.

<table><tr><td>Method</td><td>HOTA</td><td>DetA</td><td>AssA</td><td>MOTA</td><td>IDF1</td></tr><tr><td colspan="6">Tracking-by-detection</td></tr><tr><td>CenterTrack [[]</td><td>52.2</td><td>53.8</td><td>51.0</td><td>67.8</td><td>64.7</td></tr><tr><td>FairMOT []</td><td>59.3</td><td>60.9</td><td>58.0</td><td>73.7</td><td>72.3</td></tr><tr><td>DeepSORT []</td><td>61.2</td><td>63.1</td><td>59.7</td><td>78.0</td><td>74.5</td></tr><tr><td>SORT [0]</td><td>63.0</td><td>64.2</td><td>62.2</td><td>80.1</td><td>78.2</td></tr><tr><td>ByteTrack []</td><td>63.1</td><td>64.5</td><td>62.0</td><td>80.3</td><td>77.3</td></tr><tr><td>OC-SORT []</td><td>63.2</td><td>63.2</td><td>63.4</td><td>78.0</td><td>77.5</td></tr><tr><td>C-BIoU []</td><td>64.1</td><td>64.8</td><td>63.7</td><td>81.1</td><td>79.7</td></tr><tr><td>MotionTrack [B]</td><td>65.1</td><td>65.4</td><td>65.1</td><td>81.1</td><td>80.1</td></tr><tr><td>SparseTrack [4]</td><td>65.1</td><td></td><td></td><td>81.0</td><td>80.1</td></tr><tr><td>FastTracker []</td><td>66.4</td><td>66.3</td><td>66.9</td><td>81.8</td><td>82.0</td></tr><tr><td>BoostTrack++ []</td><td>66.6</td><td>一</td><td></td><td>80.7</td><td>82.2</td></tr><tr><td colspan="6">End-to-end</td></tr><tr><td>TrackFormer [四]</td><td></td><td></td><td></td><td>74.1</td><td>68.0</td></tr><tr><td>MOTR []</td><td>57.2</td><td>58.9</td><td>55.8</td><td>71.9</td><td>68.4</td></tr><tr><td>MOTRv2 []</td><td>57.6</td><td>58.1</td><td>57.5</td><td>70.1</td><td>70.3</td></tr><tr><td>MeMOTR []</td><td>58.8</td><td>59.6</td><td>58.4</td><td>72.8</td><td>71.5</td></tr><tr><td>MOTIP [日]</td><td>59.3</td><td>62.0</td><td>57.0</td><td>75.3</td><td>71.3</td></tr><tr><td>CO-MOT []</td><td>60.1</td><td>59.5</td><td>60.6</td><td>72.6</td><td>72.7</td></tr><tr><td>MOTRv3 []</td><td>60.2</td><td>62.1</td><td>58.7</td><td>75.9</td><td>72.4</td></tr><tr><td>FastTrackTr [9]</td><td>62.4</td><td>62.8</td><td>63.0</td><td>76.7</td><td>77.2</td></tr><tr><td>MOTIP2-L</td><td>60.4</td><td>63.2</td><td>58.0</td><td>76.3</td><td>72.7</td></tr><tr><td>MOTIP2-X</td><td>60.7</td><td>63.9</td><td>58.0</td><td>77.3</td><td>72.6</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 7: State-of-the-art comparison on the MOT17 [31] test set. The best end-to-end result is in bold. MOTIP2-L/X are trained on MOT17 + CrowdHuman, matching MOTIP’s recipe.

![](images/44effc47996864295d1354a4ad1f434f085c6a5487786917c062c32e12b96088.jpg)  
Figure 5: ID-switch augmentation variants. PersonPath22 (test) HOTA over joint-training epochs (MOTIP2-S): GIoU-based sampling (Ours), no augmentation (w/o IDSW), and MOTIP’s uniform permutation (uniform IDSW). The uniform permutation diverges and ends below the no-augmentation baseline, while ours trains stably to the highest HOTA.

<table><tr><td></td><td colspan="3">Training</td><td colspan="2">Inference</td></tr><tr><td>Sampling Swap%</td><td></td><td>LR%</td><td>GIoU</td><td>HOTA</td><td>LR-SW</td></tr><tr><td>DanceTrack</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>None</td><td></td><td></td><td></td><td>60.1</td><td>120</td></tr><tr><td>Uniform</td><td>37%</td><td>85%</td><td>-0.40</td><td>63.7</td><td>260</td></tr><tr><td>Ours</td><td>10%</td><td>26%</td><td>0.20</td><td>64.5</td><td>134</td></tr><tr><td colspan="6">PersonPath22</td></tr><tr><td>None</td><td></td><td></td><td></td><td>59.5</td><td>186</td></tr><tr><td>Uniform</td><td>43%</td><td>94%</td><td>-0.65</td><td>56.1</td><td>322</td></tr><tr><td>Ours</td><td>10%</td><td>31%</td><td>0.15</td><td>61.6</td><td>195</td></tr></table>

Table 8: Spatial ID-switch augmentation behavior. Training, over one epoch: the realized swap rate (Swap%), the fraction of swapped pairs that are long-range (LR%: GIoU $\leq 0 .$ non-overlapping), and their mean overlap (GIoU). Inference: the number of long-range ID switches produced by each variant (LR-sw).

![](images/cefa18069bb521359e0de3973a2cb7d3647fff880d130512594ba07ca9fb625c.jpg)  
Figure 6: A long-distance identity switch our spatial priors prevent. DanceTrack dancetrack0090, consecutive frames 710 and 711. Top: with the spatial priors disabled, identity 2 jumps ≈975 px across the frame (yellow arrow), merging two distant dancers into a single track (Sec. 1). Bottom: our full model keeps both identities stable.

## E Tracklet Confirmation Sweep

Table 9 reports the full tracklet-confirmation sweep summarized in Sec. 4.5. Requiring $N _ { \mathrm { i n i t } }$ consecutive frames of consistent assignment before committing a tentative track leaves HOTA essentially unchanged, with only slight gains in IDF1 and MOTA, while steadily reducing identity switches, by 12.5% on PersonPath22 (5612 → 4911) and 28.3% on DanceTrack $( 3 2 8 7  2 3 5 6 )$ from $N _ { \mathrm { i n i t } } { = } 1$ to 3. Accuracy plateaus at $N _ { \mathrm { i n i t } } { = } 2$ , and identity switches fall only marginally beyond it; we adopt $N _ { \mathrm { i n i t } } { = } 2$ as the smallest threshold that captures the gain.

<table><tr><td></td><td colspan="7">PersonPath22</td><td colspan="6">DanceTrack</td></tr><tr><td> $N _ { \mathrm { i n i t } }$ </td><td>HOTA</td><td>DetA</td><td>AssA</td><td>MOTA</td><td>IDF1</td><td>IDSW↓</td><td>HOTA</td><td>DetA</td><td>AssA</td><td>MOTA</td><td></td><td>IDF1</td><td>IDSW↓</td></tr><tr><td>1</td><td>61.5</td><td>63.7</td><td>60.0</td><td>72.5</td><td>70.3</td><td></td><td>5612</td><td>64.3</td><td>73.7</td><td>56.4</td><td>81.9</td><td>67.7</td><td>3287</td></tr><tr><td>2</td><td>61.6</td><td>63.8</td><td>60.3</td><td>72.9</td><td>70.4</td><td></td><td>5069</td><td>64.5</td><td>74.0</td><td>56.5</td><td>82.9</td><td>68.8</td><td>2489</td></tr><tr><td>3</td><td>61.6</td><td>63.8</td><td>60.3</td><td>73.1</td><td></td><td>70.4</td><td>4911</td><td>64.5</td><td>74.2</td><td>56.3</td><td>83.2</td><td>68.3</td><td>2356</td></tr></table>

Table 9: Effect of tracklet confirmation $N _ { \mathrm { i n i t } }$ . MOTIP2-S on PersonPath22 test and Dance-Track val. $N _ { \mathrm { i n i t } }$ is the number of consecutive frames a new tracklet must be matched before being confirmed. Best per metric and benchmark in bold.

## F Latency Comparison with Baselines

Tab. 10 compares per-frame latency against end-to-end baselines. MOTIP2 and MOTIP are profiled in PyTorch FP32 on a single V100; the FastTrackTr and PuTR figures (marked <sup>†</sup>) are estimated from the RTX 4090 measurements reported by [19], scaled by MOTIP’s V100/4090 latency ratio, so we read these cross-GPU comparisons as indicative rather than exact. At matched accuracy MOTIP2-S runs at 27.1 ms against MOTIP’s 86.9 ms (3.2× faster) while reaching higher HOTA (64.5 vs 64.0), and the lighter MOTIP2-N drops to 18.8 ms.
<table><tr><td>Method</td><td>HOTA Latency (ms) Resolution</td><td></td><td></td></tr><tr><td>End-to-end FastTrackTr-R18-Dec3 []</td><td>47.4</td><td> $1 7 . 1 ^ { \dagger }$ </td><td>640×640</td></tr><tr><td>FastTrackTr-Dec3 [] PuTR []</td><td>51.2 54.5 54.8</td><td> $2 7 . 2 ^ { \dagger }$   $3 9 . 9 ^ { \dagger }$   $3 1 . 5 ^ { \dagger }$ </td><td>640×640 800×1333 640×640</td></tr><tr><td>FastTrackTr (640) [9] FastTrackTr (1333) [] MOTIP [日]</td><td>56.9 64.0</td><td>53.6† 86.9</td><td> $8 0 0 \times 1 3 3 3$   $8 0 0 \times 1 4 4 0$ </td></tr><tr><td>End-to-end (Ours)</td><td></td><td></td><td></td></tr><tr><td>MOTIP2-N</td><td>59.2</td><td>18.8</td><td>800×1440</td></tr><tr><td>MOTIP2-S</td><td>64.5</td><td>27.1</td><td>800×1440</td></tr><tr><td>MOTIP2-M</td><td>67.3</td><td>39.3</td><td>800×1440</td></tr><tr><td>MOTIP2-L</td><td>68.0</td><td>54.8</td><td> $8 0 0 \times 1 4 4 0$ </td></tr><tr><td>MOTIP2-X</td><td>70.1</td><td>81.1</td><td> $8 0 0 \times 1 4 4 0$ </td></tr></table>

Table 10: Latency comparison with end-to-end baselines. HOTA on DanceTrack-va and per-frame latency (ms) in PyTorch on a single V100. <sup>†</sup> Estimated from RTX 4090 measurements [19] scaled by MOTIP’s V100/4090 latency ratio.