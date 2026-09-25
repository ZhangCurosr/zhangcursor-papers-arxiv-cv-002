# Training-Free Hold-Usage Detection in Sport Climbing with Foundation Pose Models

Abu Bakar

Abdullah Aftab

Virtual University of Pakistan

Amir Hamza

Army Public School and Colleges

Islamabad, Pakistan

University of Trento, Italy

chbakar26@gmail.com

Rawalpindi, Pakistan

abdullahaftabaes98@gmail.com

Trento, Italy

amir.hamza@unitn.it

Abstract—Detecting which holds a climber uses, and when, underpins automated scoring, movement analysis, and assistive systems for sport climbing. Existing approaches train task-specific models or repurpose 2D pose estimators whose hand keypoint sits at the wrist and foot keypoint at the ankle i.e. offset from the fingertips and toes that actually contact the holds, and whose hands are occluded in roughly half of all frames. We show that a frozen, off-the-shelf pose foundation model is sufficient: using the fingertip and toe keypoints of Sapiens, a per-frame proximity test against the annotated holds, per-limb mutual exclusion, and a short temporal-persistence rule, we detect hold usage without any climbing-specific training. On the The Way Up dataset (22 videos, 10 athletes, two routes), our method reaches an event F<sub>1</sub> of 90.2% on a held-out split (89.8% under leave-one-participant-out cross validation) and 79.9% over all 22 videos at any temporal overlap, and performs best on footholds (F<sub>1</sub> 89.8% overall, 96.6% held-out). Under an identical protocol it exceeds our reproductions of the YOLOv8-pose and ViTPose pipelines at every temporal threshold, with the margin widening under strict timing. An ablation shows that two intuitively helpful additions—dense foundation-feature change gating and body-part segmentation—both hurt, arguing that a minimal, keypoint-only design is the right one for this task. Finally, standard coaching statistics computed from our automatic predictions track ground truth closely (Pearson r = 1.00 for climb time, 0.94 for pace), turning ordinary single-camera video into reliable performance metrics with no instrumentation.

Index Terms—sport climbing, climbing hold detection, hold usage detection, human pose estimation, foundation models, training-free, zero-shot, keypoint detection, occlusion handling, temporal analysis, movement analysis

## I. INTRODUCTION

Climbing has grown rapidly, from its Olympic debut and expansion to the proliferation of indoor gyms and youth programs. This momentum has created real demand for tools that analyze climbs automatically for scoring competitions, giving athletes technique feedback, generating gym analytics, and powering assistive systems for visually impaired climbers [1]– [3]. Almost all of these applications rest on the same primitive: knowing which holds an athlete uses, and when.

Today, this information is obtained manually or via instrumented walls, neither of which scales. Maschek et al. formalized a video-based alternative with The Way Up dataset, casting hold-usage detection as a 2D-pose problem [4]. Their baseline uses wrist and ankle keypoints, requiring an inflated region of interest to overlap holds. This inherits two weaknesses:

a localisation problem (climbers contact holds at fingertips and toes, not wrists and ankles) and an occlusion problem (hands disappear in roughly half of all annotated frames, degrading the wrist signal). We show a frozen, off-the-shelf human-pose foundation model addresses both problems without climbing-specific training. Sapiens [5] natively predicts 308 keypoints, including individual fingers and toes. By taking these true contact keypoints directly, scoring their proximity to annotated holds, resolving per-limb ties, and applying a temporal-persistence rule, we detect hold usage accurately. The fingertip and toe keypoints make inflated regions unnecessary, and the temporal pass bridges brief occlusions. Counterintuitively, adding richer foundation cues, such as dense feature-change gating or body-part segmentation, actually degrades performance. The dominant error source is keypoint localization, rewarding a minimal, keypoint-only design.

We evaluate on The Way Up (22 videos, 10 athletes, two routes) using the dataset’s task and temporal-IoU metric. Our contributions are:

• A training-free hold-usage detector built from frozen foundation models (a promptable segmenter [6], [7] for the climber box and Sapiens [5] for pose), generalizing to unseen environments.

• A same-protocol comparison against YOLOv8-pose and ViTPose pipelines, and an ablation showing that feature gating and body-part segmentation degrade performance while minimalism wins.

• An occlusion-stratified evaluation quantifying performance in heavily occluded scenarios, which prior work highlights as hardest but does not measure.

## II. RELATED WORK

Vision for climbing. Camera-based systems analyze climbing using custom detectors or generic pose models for competition scoring, technique evaluation, and speed climbing [1], [8]–[12]. Reviews of sensor and camera methods [13], [14] show that prior works either train task-specific detectors or use generic wrist and ankle keypoints. None exploit dense fingertip or toe keypoints, and none operate training-free.

Pose-based hold-usage detection. Closest to our approach, Maschek and Schedl [4] introduced The Way Up and evaluated baseline models [15]–[17] that use wrist and ankle keypoints to define an area of interest, triggering usage after 0.5 s of overlap. Because ankle-only models place the keypoint far from the toe, they require large and imprecise foot regions, which the authors identify as a major error source. We adopt their dataset and metric but replace wrist and ankle keypoints with fingertips and toes, add per-limb mutual exclusion, and require no training.

Foundation models. Promptable segmentation [6], [18] and self-supervised features [19] generalize zero-shot across domains. Sapiens [5] natively predicts 308 keypoints, including the fingers and toes absent from COCO-style skeletons. We use these models frozen. A promptable segmenter supplies the climber box and Sapiens supplies the contact keypoints with no climbing adaptation.

Climbing datasets. Prior datasets target 3D motion, gaze, or speed climbing [8], [12], [20]. The Way $U p$ [4] is the only dataset annotating which limb uses which hold over specific frame intervals, making it the natural benchmark for our task.

## III. METHOD

We detect hold usage training-free using frozen foundation models. The input is a video and hold bounding boxes; the output is a set of usage events defining the limb, hold, and frame interval. Fig. 1 shows the pipeline. We process every second frame to keep inference cheap, relying on our temporal pass to bridge any short gaps.

Climber localization. We run a promptable segmenter [6] with a person prompt, keeping the largest mask’s bounding box. This confines pose estimation to the athlete, eliminating background distractions and bystander detections. Any zero-shot detector could serve this role.

Fingertip and toe pose. Sapiens-1B [5] runs top-down on the climber box, predicting 308 keypoints. We keep only the true contact points: the fingertip of each hand and the big toe tip of each foot. Reading these directly is our central design choice. Earlier works use wrists and ankles, which sit far from the contact point and require an enlarged, imprecise region of interest. Using fingertips and toes removes this offset entirely, making such regions unnecessary.

Per-frame contact score. For limb ℓ and hold h at frame $t ,$ we convert their distance into a score in [0, 1]:

$$
\begin{array} { r } { s _ { \ell } ^ { h } ( t ) = \mathrm { c l i p } \Big ( 1 - \frac { d ( \mathrm { t i p } _ { \ell } ( t ) , B _ { h } ) } { \tau } , ~ 0 , ~ 1 \Big ) , } \end{array}\tag{1}
$$

where $\mathrm { t i p } _ { \ell } ( t )$ is the tip keypoint, $B _ { h }$ the hold box, $d ( \cdot , \cdot )$ the point-to-box distance, and $\tau \ \mathrm { ~ a ~ }$ margin in pixels. The score is 1 inside the hold and decays to 0 at distance $\tau ,$ controlling forgiveness for localization errors. A per-frame mutual exclusion rule assigns each limb only its highest-scoring hold, removing spurious ties on densely packed routes.

Temporal aggregation. Single-frame decisions are noisy due to pose jitter, brief occlusions, or hands brushing past holds. We convert the sequence $\{ s _ { \ell } ^ { h } ( t ) \}$ into clean intervals via a four-stage temporal pass: (1) smoothing with a moving average to suppress spikes; (2) hysteresis with separate enter and exit thresholds to prevent boundary toggling; (3) gap bridging to merge intervals separated by few frames, preventing occlusions from splitting a grasp; and (4) dwell filtering to discard intervals shorter than a threshold, removing transient contacts [4]. Surviving intervals become our predicted usages $( \ell , h , t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } )$ . This linear-time pass costs almost nothing compared to pose inference.

Minimalism over richer cues. We tested two appearance-based cues: an occlusion-robust gate using dense feature changes against an empty-wall clean plate [19], and an overlap test using Sapiens’ body-part segmentation. Both lowered accuracy (table III). The task rewards a minimal design because keypoint localization, not appearance or semantics, is the dominant source of uncertainty. Adding appearance modelling introduces new failure modes (e.g., suppressing real contacts if visual changes are subtle, or failing at image scales where hands vanish) without solving the localization bottleneck. Therefore, our final model relies strictly on keypoint proximity.

## IV. EVALUATION PROTOCOL

Data and task. We evaluate on The Way Up [4]. The dataset contains 22 videos, formed from 11 participant recordings of 10 athletes climbing two routes, an easier “orange” route graded 4c and a harder “green” route graded 5a, on a vertical indoor wall. The footage was captured at 50 FPS and is provided at a resolution of $7 2 0 \times 1 2 8 0$ and a frame rate of 25 FPS. Every video comes with two kinds of annotation. The first is a bounding box for each hold on the route. The second is a list of ground truth usages, where each usage records the limb involved, the hold it used, the start and end frames of the contact, and any frame ranges in which that limb is more than 50% occluded. We take the provided hold boxes as given and our task is to detect the usages.

Split and hyperparameters. To be sure we never tune on the data we report on, we set aside participant p1, across both routes, as a dedicated validation split. Every hyperparameter, ours as well as every baseline’s, is chosen on $\mathrm { p 1 }$ alone and then frozen, and we report results on the remaining 20 videos. The selection criterion is the mean $F _ { 1 }$ over the three temporal IoU thresholds defined below, again measured on $\mathtt { p l }$ . The frozen values for our method are a margin of $\tau = 2 0 \mathrm { p x }$ , a dwell threshold of 20 frames, and hysteresis enter and exit levels of 0.4 and 0.15.

Protocols. Reporting a single split invites two opposite objections. A held out split can be accused of wasting most of an already small dataset, while tuning on all of the data, even without training, risks picking parameters that happen to flatter the test videos. To answer both at once, we report three protocols and show that they agree. The first is Held out 20, where we tune on $\mathrm { p 1 }$ , freeze, and test on the other 20 videos. This is the clean generalization number and we treat it as our primary result. The second is $L O P O { \mathrm { , } }$ , a leave one participant out cross validation in which we tune on all participants but one and test on the participant left out, repeating this so that every video is tested exactly once with parameters that were never fit to it. This is the most rigorous way to use the full dataset. The third is All 22, p1 fixed, where we simply apply the $\mathtt { p l }$ frozen parameters to all 22 videos. This is a convenient

climber box

usage events (ℓ, h, t<sub>s</sub>, t<sub>e</sub>)  
top-down pose fingertips + toes  
![](images/25f6e48340dd345ef9a48d575436b18c44f4be6ad6e065c0f954137d10bdf998.jpg)  
frozen segmenter  
video frame

![](images/c5dfde74fef3a0659a79964f2d3b4232a8b8e7a95d79213b3688a9a38feabf9f.jpg)

frozen Sapiens-1B  
![](images/a0483ed398071e1385bb6334fd67e2e1d1735f749a67d6381f15a1485070bb28.jpg)

![](images/5c4e27f90004d3a4ee7ef107b5c97ada2b53ed44f7a132d44b5014c2a49bed73.jpg)  
Fig. 1: Training-free hold-usage pipeline. Snowflakes () mark the frozen, off-the-shelf foundation models, which receive no climbing-specific training. A promptable segmenter [6] returns the climber box, Sapiens [5] estimates top-down pose with 308 whole-body keypoints of which we keep the fingertips and toes, and the proximity of each tip to the given hold box (eq. (1)), after per-limb mutual exclusion and a short temporal pass, yields the usage events.

full dataset number, though it is mildly optimistic on p1 since those videos helped choose the parameters. The three protocols agree to within 0.4 $F _ { 1 }$ (table V), which indicates that nothing is overfit to the validation participant. We therefore use the held out number as the headline result and the all 22 number (table I) for the finer grained breakdown by route and limb. Metrics. We score at the level of usage events rather than individual frames. A predicted usage and a ground truth usage are eligible to match only when they involve the same limb and the same hold. Among eligible pairs, we measure temporal overlap with the temporal Intersection over Union,

$$
\mathrm { t I o U } = { \frac { | T _ { p } \cap T _ { g } | } { | T _ { p } \cup T _ { g } | } } ,
$$

where $T _ { p }$ and $T _ { g }$ are the predicted and ground truth frame intervals [21]. Matching is greedy and ordered by score. We sort the predicted events by confidence and assign each one to the unmatched ground truth event of the same limb and hold with which it has the highest tIoU, provided that overlap clears the threshold, so that every event is used at most once. A predicted event that finds a match is a true positive, a predicted event that does not is a false positive, and a ground truth usage left unmatched is a false negative. We exclude true negatives, because the vast majority of holds in any video are never used and counting them would inflate every score [4]. From these counts we report Precision, defined as TP/(TP + FP), Recall, also called sensitivity and defined as TP/(TP + FN), and their harmonic mean $F _ { 1 }$ . We report all three at a tIoU above 0, meaning any temporal overlap at all, which is the dataset paper’s primary setting, and also at the stricter thresholds of

(2)

0.3 and 0.5, together with the mean tIoU over matched events as a direct measure of timing quality. We give these metrics overall, split into hands and feet, and split by route, and we macro average over videos so that long climbs do not dominate the score. Finally we report occlusion stratified frame recall, that is, recall computed separately over the ground truth frames flagged as occluded and those flagged as visible, which directly quantifies performance in the regime the dataset identifies as hardest but never measures.

Baselines. We reproduce the dataset paper’s keypoint method under our own evaluation so that the comparison is fair. In that method, each model’s wrist or ankle keypoint defines an area of interest, which is a box centered on the wrist for hands and a box extended downward for feet, made larger when only the ankle keypoint is available. The frame is then cropped to the route region, keypoints are stored across frames so that a momentary detection failure does not break tracking, and a hold is counted as used once the area of interest overlaps it for at least 0.5 s. We use the same two backbones as the dataset paper, namely YOLOv8-pose [17] with its ankle keypoints and ViTPose-L [16] with its coco 25 toe keypoints, and we tune each baseline’s area of interest margin and overlap thresholds on p1 under exactly the procedure described above. Our reproductions land below the numbers published in the dataset paper, because we do not replicate their exact per model margin tuning. The comparison we draw is therefore a same conditions one, in which our method and the baselines are evaluated identically, and we cite the published numbers only for context (table II).

## V. RESULTS

Across all 22 videos at tIoU > 0, our pipeline reaches an event $F _ { 1 }$ of 79.9%, with a Precision of 68.5 and a Recall of 97.2, and it reaches 90.2% on the held out 20 videos under the frozen $\mathrm { p 1 }$ parameters. The accuracy breaks down sharply by limb. Footholds are detected best, at an $F _ { 1 }$ of 89.8% over all videos and 96.6% on the held out split, while handholds are lower but still strong at 72.7%. This pattern mirrors the hand and foot gap reported for prior models, and it largely closes it, because the toe keypoints localize feet very precisely whereas hands stay harder on account of occlusion. The route comparison follows the same logic as the dataset paper, where the easier green route is detected better than the orange one, at an $F _ { 1 }$ of 84.9 against 74.9 overall, the orange route being harder because its holds sit close together. Recall stays very high throughout, at 97.2% overall and 98.9% for feet, so the limiting factor for our method is over prediction.

<table><tr><td></td><td></td><td colspan="3">orange (4c)</td><td colspan="3">green (5a)</td><td colspan="3">both (n=22)</td></tr><tr><td>tIoU</td><td>metric</td><td>overall hand foot</td><td></td><td></td><td></td><td>overall hand foot</td><td></td><td>overall hand foot</td><td></td><td></td></tr><tr><td rowspan="4">&gt; 0</td><td>R (sens.)</td><td>97.3</td><td>95.3</td><td>99.4</td><td>97.0</td><td>95.8</td><td>98.4</td><td>97.2</td><td>95.6</td><td>98.9</td></tr><tr><td>P</td><td>61.2</td><td>49.5</td><td>81.2</td><td>75.7</td><td>70.3</td><td>84.6</td><td>68.5</td><td>59.9</td><td>82.9</td></tr><tr><td> $F _ { 1 }$ </td><td>74.9</td><td>64.7 88.9</td><td></td><td>84.9</td><td>80.8 90.7</td><td></td><td>79.9</td><td>72.7</td><td>89.8</td></tr><tr><td>mean tIoU</td><td>72.5</td><td></td><td></td><td>77.8</td><td></td><td></td><td>75.2</td><td></td><td></td></tr><tr><td rowspan="5">≥ 0.3</td><td>R (sens.)</td><td>94.0</td><td>89.2 99.4</td><td></td><td>94.4</td><td>91.4 98.0</td><td></td><td>94.2</td><td></td><td>90.3 98.7</td></tr><tr><td>P</td><td>59.2</td><td>46.2</td><td>81.2</td><td>73.7</td><td>67.1</td><td>84.2</td><td>66.5</td><td>56.7</td><td>82.7</td></tr><tr><td> $\mathbf { \Pi } _ { F _ { 1 } } ^ { \mathbf { r } }$ </td><td>72.4</td><td>60.4 88.9</td><td></td><td>82.6</td><td>77.090.3</td><td></td><td>77.5</td><td>68.7</td><td>89.6</td></tr><tr><td>mean tIoU</td><td>75.4</td><td></td><td></td><td>79.9</td><td></td><td></td><td>77.6</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4"> $\geq 0 . 5 \begin{array} { l } { \mathrm { ~ P ~ } } \\ { F _ { 1 } } \end{array}$ </td><td>R (sens.)</td><td>86.0</td><td>76.5</td><td>96.4</td><td>89.5</td><td></td><td>84.6 95.0</td><td>87.8</td><td>80.5</td><td>95.7</td></tr><tr><td></td><td>54.3</td><td>40.0</td><td>78.6</td><td>69.9</td><td>62.3</td><td>81.9</td><td>62.1</td><td>51.1</td><td>80.3</td></tr><tr><td></td><td>66.3</td><td>52.1</td><td>86.1</td><td>78.3</td><td></td><td>71.4 87.7</td><td>72.3</td><td></td><td>61.886.9</td></tr><tr><td>mean tIoU</td><td>78.6</td><td></td><td></td><td>82.2</td><td></td><td></td><td>80.4</td><td></td><td></td></tr></table>

Table I. Quantitative Results. Event-level Precision, Recall, and $F _ { 1 }$ at three temporal-IoU thresholds, split by route and by hand/foot and averaged over all 22 videos using frozen p1 parameters. Footholds are detected most accurately and degrade least under strict timing, while the easier green route consistently outperforms the dense orange route.

Timing degradation. Performance degrades gracefully as the timing requirement tightens. The overall $F _ { 1 }$ is 79.9% at tIoU $>$ 0, 77.5% at $\mathrm { { t I o U } \geq 0 . 3 }$ , and 72.3% at tIoU ≥ 0.5. The foot $F _ { 1 }$ barely moves across these thresholds, going from 89.8 to 89.6 to 86.9%, whereas the hand $F _ { 1 }$ falls more steeply, from 72.7 to 68.7 to 61.8%, which reflects how much harder it is to localize an occluded hand in time. The mean tIoU over matched events actually rises as the threshold tightens, from 75.2 to 80.4, simply because only the well aligned detections survive the stricter cutoff.

Comparison with baselines. Under identical evaluation, and with every hyperparameter tuned on the same validation participant, our method beats both reproduced baselines at every threshold. On the held out 20 videos at tIoU > 0, we reach 90.2% against 71.4% for YOLOv8-pose and 67.2% for ViTPose-L. The gap widens as the timing requirement tightens, and at $\mathrm { t I o U } \geq 0 . 5$ we reach 83.2% against 57.7% and 39.9%,

which reflects the much better temporal alignment that fingertip and toe precision gives us.
<table><tr><td rowspan="2">Method (same protocol)</td><td colspan="3">event  $F _ { 1 }$  (%), held-out 20</td></tr><tr><td>tIoU &gt; 0</td><td> $\geq 0 . 3$ </td><td>≥ 0.5</td></tr><tr><td>YOLOv8-pose (ankle) [17]</td><td>71.4</td><td>66.8</td><td>57.7</td></tr><tr><td>ViTPose-L (coco_25) [16]</td><td>67.2</td><td>61.0</td><td>39.9</td></tr><tr><td>Ours (fingertip/toe)</td><td>90.2</td><td>88.7</td><td>83.2</td></tr><tr><td colspan="4">Context: published ViTPose-L acc. 86.6% [4].</td></tr></table>

Table II. Same-protocol comparison. All methods tuned on p1 and evaluated on the held-out 20 videos. Our reproductions of baselines fall below the published results; under identical conditions our fingertip/toe method wins at every threshold, with the gap widening under strict timing thresholds.

The baselines show the high recall and low precision profile one would expect, since their wrist and ankle regions of interest fire on several neighbouring holds at once, most of all on the densely packed orange route. For context, the dataset paper reports a published ViTPose accuracy of 86.6% at tIoU > 0 [4]. Our same conditions reproductions sit below this because we do not replicate their per model margin tuning, which is exactly why we compare against our own reproductions rather than their published values.

Why the baselines fall behind. The two reproduced baselines share that high recall and low precision profile for a structural reason. Both anchor their region of interest at the wrist or the ankle and then inflate it to reach the real contact point, and this is worst for YOLOv8-pose, whose ankle only keypoint forces a large downward box to cover the toe [4]. On the orange route, where holds sit close together, such a box overlaps several holds at once, so the method reports several simultaneous usages and precision collapses, which is why our reproduced ViTPose-L manages only 39.9% $F _ { 1 }$ at $\mathrm { { t I o U } \geq 0 . 5 }$ . Fingertip and toe keypoints make the region unnecessary, because the contact point is the keypoint itself, so exactly one hold is selected per limb and the per frame mutual exclusion rule settles the rare tie. The same mechanism explains why our advantage grows with the temporal threshold. Precise contact points produce intervals whose start and end line up with the true grasp, whereas an inflated box switches on too early and off too late.

<table><tr><td>Configuration</td><td> $F _ { 1 }$  @ tIoU &gt; 0 (%)</td></tr><tr><td>Body-part segmentation only</td><td>1.1</td></tr><tr><td>+ dense feature-change gate</td><td>80.4</td></tr><tr><td>Fingertip/toe proximity (final)</td><td>90.2</td></tr></table>

Table III. Ablation (held-out 20). Two intuitively helpful additions both degrades the performance: body-part segmentation is unusable at the scale Sapiens requires, and the dense feature change gate suppresses true usages.

Error analysis. Our errors concentrate exactly where the pose is least certain. Recall is uniformly high (table I), so missed usages are rare, and the dominant error is over prediction. It is worst for hands on the orange route, where precision drops to

49.5% at tIoU > 0, because the short and closely spaced moves there, combined with frequent hand self occlusion, produce brief spurious contacts on neighbouring holds. Feet stay easy throughout, with a foot $F _ { 1 }$ of at least 86.9% at every threshold, because toes are rarely occluded and footholds are better separated on the wall. Near the top of the route the climber occupies a small part of the frame and pose confidence drops, which produces the short false positives visible in the qualitative timelines. None of these failure modes is one that appearance cues would fix, because they all stem from keypoint uncertainty under occlusion and small scale, and this is consistent with the ablation result that richer cues do not help.
<table><tr><td>Frame recall (%)</td><td>orange</td><td>green</td><td>both</td></tr><tr><td>Occluded GT frames (&gt; 50%)</td><td>75.2</td><td>86.8</td><td>81.0</td></tr><tr><td>Visible GT frames</td><td>97.0</td><td>98.9</td><td>97.9</td></tr></table>

Table IV. Occlusion-stratified frame recall. Temporal persistence maintains usage through most occluded spans with no explicit occlusion model; occlusion (mostly of hands) remains the hardest regime, consistent with the dataset.

Ablation study. Starting from the full keypoint model, the two richer cues we implemented both reduce accuracy. Body part segmentation on its own is almost useless, at an $F _ { 1 }$ of 1.1%, because at the whole image scale that Sapiens needs, the climber’s hands and shoes are simply too small to segment reliably. The dense feature change gate, which we intended as a way to add occlusion robustness, instead lowers $F _ { 1 }$ from 90.2% to 80.4%, because it suppresses genuine usages during contact. Per frame mutual exclusion leaves the score essentially unchanged at our tuned operating point, while removing the densely packed false positives that begin to matter at the stricter thresholds. The final model therefore relies on fingertip and toe proximity alone.

Occlusions. Splitting frame level recall by the dataset’s occlusion flags, we recover 81.0% of the occluded usage frames overall, broken down as 75.2% on the orange route and 86.8% on the green route, against 97.9% of the visible frames. Occlusion remains the hardest regime, which agrees with the dataset paper, but the temporal persistence and gap bridging in our method keep a usage alive through most occluded spans without any explicit model of occlusion.

Reporting protocol. Our headline number is stable across the way the data is split. The held out result of 90.2%, the all 22 result with frozen parameters of 90.0%, and the LOPO cross validation result of 89.8% all agree to within 0.4 $F _ { 1 }$ , which indicates that the hyperparameters chosen on p1 do not overfit it.

Sensitivity and timing. The method is not delicately tuned. The $F _ { 1 }$ is essentially flat across fingertip margin values from 20 to 60 px and rises smoothly with the dwell threshold, so neither setting needs to be hit precisely. All models are frozen and run per frame, and the inference cost is dominated by the pose backbone.

<table><tr><td>Protocol</td><td>headline  $F _ { 1 }$  (%)</td></tr><tr><td>Held-out 20 (val = p1)</td><td>90.2</td></tr><tr><td>All 22, frozen p1 params</td><td>90.0</td></tr><tr><td>LOPO (each video tested once)</td><td>89.8</td></tr></table>

Table V. Protocol stability at tIoU >0. The three honest reporting protocols agree to within 0.4 $F _ { 1 }$ , indicating the hyperparameters chosen on the validation participant do not overfit it.

## VI. QUALITATIVE RESULTS

Usage timelines. A per-(ℓ, h) timeline that overlays predicted usages against ground truth, colour-coded by TP/FP/FN, shows predictions tracking the true bottom-to-top progression of the climb. Most usages are recovered with start and end frames close to the annotation, and the few errors are short: occasional false positives near the top of the wall, where the climber occupies a small fraction of the frame and pose confidence is lowest. The timelines also make the route difference visible— on the densely-packed orange route, hand tracks show more short spurious segments on adjacent holds than on the green route, matching the lower hand precision in table I.

Per-frame overlays. Frame-level overlays that draw the detected fingertips and toes together with the hold each limb is assigned confirm the central claim of the method: the contact keypoints land on the correct holds even when the wrist or ankle is laterally offset or the foot is pointed, exactly the situations that force wrist/ankle baselines to enlarge their region of interest. During hand self-occlusion—where the hand disappears behind the torso for a span of frames—the gap-bridging stage keeps the correct hold assigned across the occluded interval rather than terminating and restarting the usage, which is what preserves recall on occluded frames (table IV).

Feature visualization. A PCA of the dense foundation features, and of their change against a climber-free clean plate, separates the climber cleanly in feature space. This clean separation is what motivated our feature-change gate experiment; the ablation, however, shows the signal does not improve detection and in fact suppresses genuine contact (table III), so we report the visualization only as a qualitative observation rather than as a component of the method.

## VII. APPLICATIONS: COACHING STATISTICS FOR FREE

Hold usage is the atomic unit of a climb. Once we know which hold each limb used and when, an ascent stops being raw video and becomes a structured sequence of events that can be measured, summarized, and compared across climbers. This structure is what scoring, technique analysis, gym analytics, and assistive next hold guidance all ultimately rely on. We show that the structure is immediately useful by deriving a set of standard climbing performance statistics directly from our predicted usage events, with no additional model to train and no extra annotation to collect.

a) Statistics.: From the event stream alone we compute six quantities that coaches care about. The total climb time is the span from the first contact to the last release. The number of moves is simply the count of usage events. The foot usage ratio is the fraction of those events that are feet, and it matters because loading the legs rather than hanging on the arms is a hallmark of efficient technique, so a higher ratio tends to indicate a stronger climber. The mean dwell per hold is the average duration of a usage, which serves as a proxy for hesitation and for forearm pump, since a tired or uncertain climber lingers on each hold. The pace is the number of moves per second, a measure of fluency. The mean limb flight time is the average gap between releasing one hold and contacting the next, which captures how crisp the movement is. Because dwell is localized to individual holds in time, the holds with the longest dwell also pick out the crux of the route, the section where climbers pause to plan their next move or to recover.

b) Discriminative power.: These statistics separate the ten athletes cleanly. Across the dataset, climb time ranges from 35 to 186 s, mean dwell from 3 to 11 s, and foot usage from 36 to 58%, which is more than enough spread to tell a fast, fluent, leg driven ascent apart from a slow, hesitant, arm dependent one. These are exactly the quantities a coach would otherwise extract by hand from the footage, and here we obtain them automatically from a single video.

c) Agreement with ground truth.: A statistic is only useful if it survives the move from ground truth events to our predicted events. On the held out climbs, the statistics we compute from our automatic predictions track those computed from the annotations very closely, with a Pearson correlation of r = 1.00 for total time, 0.94 for pace, 0.93 for mean dwell, and 0.84 for foot usage ratio. Total time is essentially exact because it depends only on the first and last events of the climb, which are the easiest to get right. Foot usage is the loosest of the four because it inherits the residual hand false positives discussed in the error analysis, yet a correlation of 0.84 is still strongly indicative. Taken together, these results show that the training free pipeline turns an ordinary single camera recording into reliable coaching metrics, with no instrumented wall, no body worn sensors, and no training specific to a given gym.

## VIII. CONCLUSION

We presented a training-free method for hold-usage detection in sport climbing that uses the fingertip and toe keypoints of a frozen pose foundation model with a simple proximity and temporal-persistence rule. It reaches 90.2% event $F _ { 1 }$ on heldout data (89.8% under cross-validation), outperforms sameprotocol reproductions of YOLOv8-pose and ViTPose, and is most accurate on footholds. An ablation shows that richer foundation-model cues are unnecessary and even harmful, arguing for minimal designs on this task.

Limitations .We assume the holds bounding boxes are given, as provided by the dataset, and detect usage rather than the holds themselves; integrating zero-shot hold detection is future work. Hands under prolonged occlusion remain the dominant error source. Our reproduced baselines fall short of their published accuracy, so comparisons should be read as same-conditions, not as claims of accuracy state of the art over the published ViTPose numbers. Finally, the dataset covers a single vertical wall and viewpoint; broader wall angles and camera placements remain to be evaluated.

## REFERENCES

[1] M. Michenthaler, “Automated scoring in climbing competitions,” Master’s thesis, TU Wien, 2022.

[2] J. Boulanger, L. Seifert, R. Herault, and J.-F. Coeurjolly, “Automatic ´ sensor-based detection and classification of climbing activities,” Sensors, 2016.

[3] M. Richardson, K. Petrini, and M. Proulx, “Climb-o-vision: A computer vision driven sensory substitution device for rock climbing,” in Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, 2022.

[4] A. Maschek and D. C. Schedl, “The way up: A dataset for hold usage detection in sport climbing,” in CVPRW, 2025.

[5] R. Khirodkar, T. Bagautdinov, J. Martinez, S. Zhaoen, A. James, P. Selednik, S. Anderson, and S. Saito, “Sapiens: Foundation for human vision models,” in ECCV, 2024.

[6] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollar, and R. Girshick,´ “Segment anything,” in ICCV, 2023.

[7] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle,¨ C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, and C. Feichtenhofer, “SAM 2: Segment´ anything in images and videos,” in ICLR, 2025.

[8] H. Vrzakov ´ a, J. Koskinen, S. Andberg, A. Lee, and M. J. Amon, “Towards´ automatic object detection and activity recognition in indoor climbing,” Sensors, 2024.

[9] R. Beltran, J. Richter, G. K ´ ostermeyer, and U. Heinkel, “Climbing ¨ technique evaluation by means of skeleton video stream analysis,” Sensors, 2023.

[10] Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, and Y. Sheikh, “OpenPose: Realtime multi-person 2d pose estimation using part affinity fields,” TPAMI, 2021.

[11] D. Pandurevic, P. Draga, A. Sutor, and K. Hochradel, “Analysis of competition and training videos of speed climbing athletes using feature and human body keypoint detection algorithms,” Sensors, 2022.

[12] P. Elias, V. Skvarlova, and P. Zezula, “Speed21: Speed climbing motion dataset,” in MMSports, 2021.

[13] M. Andric, F. Ricci, and F. Zini, “Sensor-based activity recognition and´ performance assessment in climbing: A review,” IEEE Access, 2022.

[14] J. Boulanger, L. Seifert, R. Herault, and J.-F. Coeurjolly, “Automatic´ sensor-based detection and classification of climbing activities,” Sensors, 2016.

[15] C. Lugaresi, J. Tang, H. Nash, C. McClanahan, E. Uboweja, M. Hays, F. Zhang, C.-L. Chang, M. G. Yong, J. Lee et al., “MediaPipe: A framework for perceiving and processing reality,” in CVPRW, 2019.

[16] Y. Xu, J. Zhang, Q. Zhang, and D. Tao, “ViTPose: Simple vision transformer baselines for human pose estimation,” in NeurIPS, 2022.

[17] G. Jocher, A. Chaurasia, and J. Qiu, “Ultralytics YOLOv8,” https://github. com/ultralytics/ultralytics, 2023.

[18] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Radle,¨ C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollar, and C. Feichtenhofer, “SAM 2: Segment´ anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024.

[19] M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, M. Assran, N. Ballas et al., “DINOv2: Learning robust visual features without supervision,” TMLR, 2024.

[20] M. Yan, X. Wang, Y. Dai, S. Shen, C. Wen, L. Xu, Y. Ma, and C. Wang, “CIMI4D: A large multimodal climbing motion dataset under humanscene interactions,” in CVPR, 2023.

[21] F. Caba Heilbron, V. Escorcia, B. Ghanem, and J. Carlos Niebles, “ActivityNet: A large-scale video benchmark for human activity understanding,” in CVPR, 2015.