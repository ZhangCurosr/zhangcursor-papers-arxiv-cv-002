# What a gated sensing pipeline never looks at: bandwidth reduction and the misses behind it

Raghu Venkat

Tricha Anjali Numberz.ai Inc., Alpharetta, GA

Draft of 13 September 2026

## Abstract

An airborne sensor on a contested link cannot send video, so the appealing move is to send findings instead and report the ratio between the two. We evaluate a gated sensing pipeline that does this, combining learned object detection and image–text comparison with deterministic scheduling, gating, evidence accumulation and transmission rules. On staged footage with the semantic stage live it sends 38,736 bits over 211 s, a reduction of 41,977×, and names 1 of 4 staged events with no false report. That detection has since been superseded: a correction to how the tracker measures speed removed the measurement artefact the normality model had been learning from, and the flight no longer warms. Under the corrected code the pipeline names 1 of 7 staged events across four flights, and we report both. On a control flight where nothing was staged it reports nothing, a reduction of 155,830×: the largest number in the study and the least informative, because a reduction ratio measures the scene.

The mechanisms that produce the reduction also decide which observations ever reach a decision, so the two cannot be reported apart. We give a tick-level trace of one flight (3,187 rows) that places each of three missed events at the stage where it stopped progressing: one produced no track, one failed the structural place test at 0.129, and one passed 539 structural ticks but reached only 3.192 against a boundary of 3.807. We also report one instance of a known failure mode, an online normality model absorbing the object it will later judge, measured against the threshold that object then failed.

The evidence is one detection and six misses across four staged flights, beside one clean control, and we treat it as a case study. We give the reproduction protocol and generated results, identify which supporting artifacts are not distributed, and state which experiments did not run.

## 1 Introduction

The bandwidth argument for on-board analysis is easy to make and easy to overstate. A downlink that cannot carry video can carry a sentence, so a system that decides on board and transmits only findings will always show a large ratio between what it saw and what it sent. The ratio is arithmetically true and it is close to uninformative on its own, because the quietest scene produces the largest number. Our own control flight demonstrates this: nothing was staged, nothing was reported, and the reduction is 155,830×, four times the ratio on the flight where something actually happened.

A maintainer of such a system needs three things. How much of the source was never examined, and by which rule. Which events the system missed, and at which stage. And whether the false-alarm behaviour holds on material where a false alarm is the sole possible outcome.

This paper reports those three for one pipeline. The pipeline combines learned object detection and image–text comparison with deterministic scheduling, gating, evidence accumulation and transmission rules. The semantic comparison contributes one weighted term to an evidence score and cannot emit a report by itself. Stage-level traces identify where each observed event ceased to progress through the pipeline; they do not establish that the first blocking stage was the sole cause of the miss.

We ofer two contributions. The first is a reporting pattern: a tick-level attribution of every missed event to the stage where it stopped progressing, over a trace of 3,187 rows, published alongside the reduction ratio the same mechanism produces. The second is one clean instance of a known failure mode, an online normality model absorbing the very object it will later be asked to judge, measured at track-and-cell granularity with the contaminated score printed against the threshold it failed.

Two further results are reported but not claimed as contributions. A coverage and compute profile with the semantic model live shows what fraction of the source reached each stage and that the regularly scheduled detector dominates the cost. A counterfactual over the same trace widens the gate and recovers nothing; Section 5.4 explains why that is close to a corollary of the contamination finding rather than an independent test.

## 2 Related work

Four bodies of work bound this one, and we claim novelty against none of their mechanisms.

Task-oriented and semantic communication asks what a link should carry when the receiver has a task instead of a fidelity requirement, and optimises the channel around downstream value instead of reconstruction error [1]. Our pipeline is an instance of that idea with the optimisation done by a declared rule. Our contribution concerns how such a system should be reported.

Edge video analytics and cascaded inference places cheap stages ahead of expensive ones so that the expensive model runs rarely; NoScope [2] is the canonical treatment, with specialised models and diference detectors deciding when the reference network is consulted. The cascade there is tuned to preserve the reference model’s accuracy. Ours cannot be, because there is no reference model to preserve: the question is not whether we reproduce a full-rate system’s answers but which events never reach a decision at all.

Event-triggered sensing and control formalises acting only when a condition fires, and studies the trade between communication and performance that follows [3]. The gate in Section 3 is an event trigger whose condition is learned online from the scene, so its failure modes have to be found by measurement.

Open-vocabulary grounding. The semantic stage is an image–text comparison against operatorwritten words, in the manner of CLIP [4]. That choice is what lets the mission change without retraining, and it is also why a miss at that stage cannot be diagnosed by inspecting a class list.

Our contribution sits across these: we evaluate a gated pipeline by accounting jointly for the communication it saves and the events each gate excluded, and we attribute every observed miss to the stage that prevented its examination.

## 3 The pipeline

Eight stages. Two carry learned models: an object detector exported to ONNX and an image encoder compared against text embeddings. The scheduling, gating, evidence accumulation and transmission rules around them are deterministic. The detector runs on a fixed schedule. The gate decides when the semantic comparison runs.

Ingest and stabilise. Frames are decoded at 15 fps; skipped frames are grabbed without decoding. Ego-motion is removed by estimating a homography from sparse optical flow on a grey frame resampled to 360 px high, with corners re-detected every 10 full estimates or whenever fewer than 80 survive RANSAC. A full estimate is skipped when a $6 4 \times 3 6$ thumbnail diference falls below 1.5, with a 2.0 s keepalive so that slow creep cannot accumulate unseen. Stabilisation runs at ingest rate, which is where the sweep in the device card leaves it: below 15 Hz the tracks thin out enough to change the result.

Detect and track. An ONNX detector with a 512 px input runs at 5 Hz, below frame rate, and its boxes are associated into tracks. This stage is the dominant cost (Section 5.5) and its rate is the schedule’s main lever.

Normality. Each track, on each tick, folds its stabilised position into a grid of visit counts, a per-cell speed mean and variance, and a per-cell dwell EMA. Three surprise scores follow. Place is $1 - ( v / v _ { \mathrm { m a x } } ) ^ { 0 . 3 5 }$ , clipped to [0, 1], where v is the cell’s visit count and $v _ { \mathrm { m a x } }$ the busiest cell’s. Speed is $| s - \mu | / 3 \sigma$ clipped, computed only where a cell has more than six visits and scaled by 1.25 when the object is slower than a locally busy norm; where local evidence is too thin the score is held at 0.5. Dwell is $( d - \bar { d } ) / 2 . 5 \bar { d }$ clipped, against the cell’s EMA. The model is warm only once it has observed enough elapsed time and a floor of observations; before that it refuses to score and says so on the link. Warm-up is elapsed-time-based because an observation count made warm-up a function of how often the stabiliser ran.

The gate. A track is eligible for semantic examination when its place score exceeds 0.55. Refusals are recorded with a reason: unchanged since last examined, too recent, over the rate ceiling, or already hopeless. The last is a scheduler rule: further semantic examination is suppressed once accumulated evidence falls below −1.5. The accumulator itself still admits positive increments, so the boundary is not unreachable from there. The examination that could produce one is what stops. An early negative assessment can therefore prevent its own reconsideration, which Section 7 returns to.

Semantic comparison. For an eligible track, one image–text comparison is made against the operator’s mission card, rate-limited to at most one per track per 4 s and two per second overall. It returns a similarity in [0, 1] and nothing else.

Evidence accumulation. With ℓ the leak per second, the score updates as $S \gets \ell ^ { \Delta t } S + \lambda , \Delta t$ in seconds. When the structural preconditions fail, $\lambda = - 0 . 2 5$ : evidence against, but weak, so a momentary occlusion does not erase minutes of watching. Otherwise

$$
\lambda \ = \ K \left( \frac { w _ { \mathrm { s e m } } c _ { \mathrm { s e m } } + w _ { \mathrm { a n o m } } c _ { \mathrm { a n o m } } } { w _ { \mathrm { s e m } } + w _ { \mathrm { a n o m } } } - \theta \right) ,
$$

with $K = 3 . 0 , \theta = 0 . 5$ , and weights $w _ { \mathrm { s e m } } = 3 . 0 , w _ { \mathrm { a n o m } } = 1 . 0$ on the edge profile: place is support and cannot veto a semantic match. The leak is $\ell = 0 . 9 8 5$ per second, so a claim must keep being re-earned and a vehicle that leaves stops being reported without a rule saying so. S is clamped two units outside each boundary, so a long quiet period cannot drive the system deaf. The boundaries are Wald’s [8] at $\alpha = 0 . 0 2$ and $\beta = 0 . 1 0 \colon$ ln ${ \frac { 1 - \beta } { \alpha } } = + 3 . 8 0 6 7$ and ln $\textstyle { \frac { \beta } { 1 - \alpha } } = - 2 . 2 8 2 4$ , quoted to three decimals elsewhere. As in related sequential work the increment is a declared linear form and not a fitted likelihood ratio, so α and $\beta$ fix where the test stops and are not claimed as operational error probabilities.

<table><tr><td>Material</td><td>Grounding</td><td>Ground truth</td><td>What it can support</td></tr><tr><td>3 synthetic clips</td><td>proxy</td><td>exact, by construction</td><td>detection and false alarms against known truth</td></tr><tr><td>Flight 0104</td><td>CLIP</td><td>staged, activity log</td><td>detection on real footage</td></tr><tr><td>Flight 0101</td><td>CLIP</td><td>nothing staged</td><td>false alarms only</td></tr></table>

Table 1: The material, and the two grounding modes. The synthetic clips ran without the semantic weights loaded and the real-footage takes ran with them. They are not two arms of one experiment and are not pooled anywhere in this paper. This table is the material Sections 5.2–5.5 are built from; Table 6 carries the full set of staged flights and every scored run of each.
<table><tr><td>Material</td><td>Duration</td><td>Sent</td><td>Reduction</td><td>Reports</td></tr><tr><td>Flight 0104, CLIP, pre-correction</td><td>211 s</td><td>38,736 bit</td><td>41,977×</td><td>1</td></tr><tr><td>Flight 0101, CLIP, nothing staged</td><td>302s</td><td>22,240 bit</td><td>155,830×</td><td>0</td></tr><tr><td>Synthetic clip 1, proxy</td><td>360 s</td><td>40,896 bit</td><td>7,032×</td><td>1</td></tr><tr><td>Synthetic clip 2, quiet, proxy</td><td>120s</td><td>8,208 bit</td><td>11,656×</td><td>0</td></tr><tr><td>Synthetic clip 3, proxy</td><td>150s</td><td>26,928 bit</td><td>4,144×</td><td>1</td></tr></table>

Table 2: Link accounting. The ratio is source bits over emitted bits. The largest ratio in the table belongs to the flight where nothing was staged and nothing was reported. The flight 0104 row is the run made before the speed-measurement correction of Section 5.7; under the corrected tracker that flight emits no report at all.

Emit. Crossing the upper boundary sends a receipt: the clause that fired, the evidence value, the dwell, a bounding box, a small image chip, and a hash chained to the previous record. Heartbeats are sent regardless, so silence is distinguishable from a dead radio.

## 4 Data and protocol

The synthetic clips are generated with a fixed seed and carry exact ground truth by construction. The two real flights are staged scenes over a car park, with ground truth taken from an activity log written at the time. Flight 0104 staged four events; flight 0101 staged none, so a false alarm is the sole possible outcome.

Both real flights are scored twice in the sense that matters: the pipeline either emitted a report matching a staged event, or it did not, and any report not matching a staged event is a false alarm. There is no partial credit and no confidence threshold to sweep.

## 5 Results

## 5.1 Link accounting

What the denominator is. Source bits are the encoded delivered file, taken as its size on disk in bytes and multiplied by eight. For the real flights that file is H.264 at $1 2 8 0 \times 7 2 0$ , nominally 30 fps, at 7.71 Mbit/s on flight 0104 and 11.47 Mbit/s on flight 0101; the two are not interchangeable and the ratio uses each flight’s own. The ratio is therefore against an already-compressed stream, the more conservative of the two choices and the one we can reproduce. No matched codec baseline accompanies these runs (Section 6), so the ratio has no external reference point and we do not present it as a comparison against anything.

Table 2 is the number a bandwidth argument would lead with, and it should not be read without Section 5.3. The ratio rises when the scene is empty, when the gate is tight, and when the system is wrong in the direction of silence. It falls when the system reports. On the control flight the pipeline achieved its best compression by finding nothing, in a scene where finding nothing was correct; the same behaviour on a scene with an event in it is a miss, and the ratio does not distinguish the two cases.

We report the ratio beside the detection outcome for the same material. There is no single headline compression figure for the system.

## 5.2 Synthetic clips: nothing matched

On 630 s of synthetic video with exact ground truth, run in proxy mode, the pipeline matched 0 of 6 staged events. It emitted 1 report on clip 1 and 1 on clip 3, neither matching a staged event. One false report in 360 s and one in 150 s normalise to 10 and 24 per hour, and we give those as observed counts scaled to an hour. They are not estimated rates: a single event per clip does not support a rate, for the same reason one clean control flight does not. On the quiet clip it emitted 0 reports and 0 false alarms, which is the correct behaviour and the only one of the three that passes its own scoring rule.

What the proxy is. It is not a null and it is not neutral, so the result cannot be read without it. The proxy scores a box from geometry and motion alone: aspect ratio and area relative to the frame give a vehicle-like and a person-like term, speed enters through a logistic centred at 3 px/s, dwell is clipped at 60 s, and the anomaly score from the normality stage is added with a small weight. It then selects one of three weightings by substring-matching the mission card’s intent phrase, so the operator’s words choose the formula but never reach an embedding. No image or text model is loaded. The manifest records semantic result: false and the runner refuses to proceed without an explicit flag.

This is the strongest negative result in the paper. With that proxy in place of the semantic comparison, the pipeline does not find the events it was built to find, and it produces alarms on material where none should occur. Because the proxy is a structural scorer and an absent stage would behave diferently, the result is consistent with two readings we cannot separate here: the pipeline failing, or the proxy failing. Running these clips with the semantic weights loaded would separate them, and Section 6 lists that as not run.

One conclusion follows and a second does not. The conclusion is that this pipeline, with a stand-in replacing the semantic comparison, is not suficient on these clips. The conclusion we do not draw is any inference about the full pipeline from these runs, because the substituted component is one of the learned components under evaluation.

## 5.3 Real footage with the semantic stage live

On flight 0104, with CLIP weights loaded and under the pre-correction tracker (Section 5.7 gives the run this came from and why the corrected code does not reproduce it), the pipeline reported 1 of 4 staged events with 0 false alarms. The report was the silver sedan stopped on the street: evidence crossed at 4.022 against the 3.8067 boundary, 47.9 s after the event began, in 23,520 bits.

<table><tr><td>Staged event</td><td>Place</td><td>Ticks</td><td>If widened</td><td>Max S</td><td>Failed at</td></tr><tr><td>Silver sedan (reported)</td><td>0.618</td><td>543</td><td>543</td><td>5.807</td><td></td></tr><tr><td>Dark car at driveway</td><td>0.702</td><td>539</td><td>539</td><td>3.192</td><td>semantic/evidence</td></tr><tr><td>Black pickup on verge</td><td>0.129</td><td>0</td><td>0</td><td>0.000</td><td>structural gate</td></tr><tr><td>Handover pair</td><td></td><td>0</td><td></td><td></td><td>detect/track</td></tr></table>

Table 3: Every staged event on flight 0104, traced tick by tick over 3,187 rows. Max S for the reported event is at the clamp, above its crossing value: S is held two units outside each boundary, so 3.8067 + 2 gives 5.807. It crossed at 4.022. Place is the mean place-rarity score; ticks are those passing the structural test as built; “if widened” is the same count under a composite gate that also credits speed and dwell surprise.

On flight 0101, where nothing was staged, it emitted 0 reports and 16 heartbeats over 302 s, at 74 bit/s. No false alarms.

One detection and three misses on this flight, beside one clean control, is a thin result and we present it as one. Table 6 carries the full flight set and Section 5.7 the number it supports. This pair is enough to establish that the pipeline runs end to end on real footage and that its silence on an empty scene is genuine and the link is alive. It is not enough to support a detection rate, and we do not quote one.

## 5.4 Three misses, three diferent stages

Table 3 carries the paper’s main result. The three misses stop at three diferent stages. They do not have three isolated causes.

The handover pair never produced a track that matched its truth box, so no later stage ever saw it. Nothing about the structural gate, semantic comparison or evidence rule is implicated; the failure occurs before those stages. The trace is per-track, so it cannot tell us whether the detector fired on the pair and association failed, or whether the detector never fired at all. We therefore attribute the loss to the detect/track stage and do not name the tracker.

The black pickup produced a track and failed the structural test, at a mean place score of 0.129 against a threshold of 0.55. It was parked across a driveway apron for the whole flight, so the scene’s own normality model had learned that a vehicle there was ordinary. This is the gate working exactly as designed and being wrong, which is the more interesting of the two possibilities.

The failure mode is not new. An adaptive background model absorbing a stationary foreground object, and then treating it as background, is long documented in background subtraction [5, 6], where it appears as foreground absorption, ghosting or the sleeping-person problem. We add an instance at track-and-cell granularity in a place-rarity score, with the contaminated value printed against the threshold it failed. The mechanism is given in full below. Every track folds its own position into the visit count of the cell it occupies, on every tick. A stationary object therefore drives up the visit count of precisely the cell it is being judged against. Inverting the place score, $0 . 1 2 9 = 1 - ( v / v _ { \mathrm { m a x } } ) ^ { 0 . 3 5 }$ puts the pickup’s cell at roughly two-thirds of the busiest cell’s visits on a flight where the pickup never moved. It made its own cell ordinary. The event of interest contaminates the definition of normal that will later be used to judge it. The pickup was present for the whole flight, so it was never anomalous by the measure the system had. Any deployment of this architecture needs something that breaks that loop: a qualified warm-up on material believed clean, a frozen reference learned once and not updated, an operator-declared mask over regions where presence is never ordinary, or a historical normality carried between sorties. None is implemented here. The pickup is the cost.

<table><tr><td rowspan="2">Stage</td><td colspan="2">Flight 0104, pre-correction</td><td colspan="2">Flight 0101</td></tr><tr><td>Calls</td><td>% time</td><td>Calls</td><td>% time</td></tr><tr><td>Stabilise</td><td>3,160</td><td>3.33</td><td>4,526</td><td>4.24</td></tr><tr><td>Detect</td><td>1,054</td><td>91.49</td><td>1,509</td><td>91.50</td></tr><tr><td>Track</td><td>1,054</td><td>2.89</td><td>1,509</td><td>3.46</td></tr><tr><td>Semantic comparison</td><td>12</td><td>2.11</td><td>7</td><td>0.71</td></tr></table>

Table 4: Per-stage call counts and share of stage time on the two real flights, with the semantic weights loaded. Flight 0104 is the pre-correction run (Section 5.7). Percentages are of total stage time and do not sum to 100 because the smaller stages are omitted.

The dark car is the one case where the semantic and evidence stages are implicated together. The accumulated score depends on the schedule, the cached value, the weights and the decay as much as on the comparison itself, and this trace cannot separate them. It passed 539 structural ticks, comparable to the 543 of the event that was reported, and its evidence reached 3.192 against the 3.807 boundary, where it stopped.

Gate counterfactual. The counterfactual column recomputes the same trace under a composite gate crediting speed and dwell surprise alongside place. The passing-tick counts are identical for every event: 543 against 543 for the sedan, 539 against 539 for the dark car, and 0 either way for the pickup. Only one of the three events was ever eligible for recovery by a gate change. The handover pair produced no track, so no gate can reach it; the dark car passed the gate 539 times already and failed later. That leaves the pickup, and it fails the composite gate for the same reason it failed the place score: its speed is zero because it is parked, and its dwell is judged against a cell average it spent the flight raising. The counterfactual is therefore close to a corollary of the contamination finding rather than an independent test of the gate. Widening’s cost in false reports is not established here either: this trace follows the staged events only.

## 5.5 Coverage and cost by stage

Table 4 gives how much of the source reached each stage.

Coverage. Every ingested frame is stabilised: 3,160 on flight 0104 and 4,526 on 0101. Detection runs on 33.4% and 33.3% of them, which is the declared 5 Hz budget against 15 fps ingest. Nothing adapts it. The semantic comparison is where the gate bites: it ran 12 times against 1,431 declined on flight 0104, a duty cycle of 0.83%, and 7 against 2,003 on flight 0101, 0.35%.

Why each refusal happened is recorded separately. On flight 0104, 1,374 of the refusals were because the track had not changed since it was last examined and 57 because its accumulated evidence was already too negative to recover. On flight 0101 the split reverses: 730 unchanged against 1,272 hopeless, and 1 for having been examined too recently, which is the whole of the 2,003. An empty scene is refused mostly for being uninteresting; a scene with something in it is refused mostly for being unchanged.

Cost. Detection dominates on both flights, at 91.49% and 91.50% of stage time. The semantic comparison costs 2.11% and 0.71%. Both stages are learned; what separates them is the schedule. The expensive one is the detector, which runs on a fixed budget, and the lever that moves the compute total is therefore the detector rate. The model the gate protects is the cheaper of the two. We report this from the real-footage runs specifically because the synthetic runs substituted the semantic stage and cannot speak to its cost.

<table><tr><td>Configuration</td><td>Mean W</td><td>Idle W</td><td>Peak W</td><td>Incr. W</td><td>Energy J</td><td>×RT</td></tr><tr><td>Run, fitted, default governor</td><td>5.33</td><td>2.83</td><td>9.85</td><td>2.49</td><td>1,379</td><td>0.81</td></tr><tr><td>Run, fitted, powersave</td><td>3.74</td><td>2.93</td><td>6.43</td><td>0.81</td><td>1,469</td><td>0.54</td></tr><tr><td>Run, removed, default governor</td><td>4.78</td><td>2.29</td><td>9.21</td><td>2.49</td><td>1,218</td><td>0.82</td></tr><tr><td>Run, removed, powersave</td><td>3.02</td><td>2.28</td><td>5.23</td><td>0.74</td><td>1,189</td><td>0.54</td></tr></table>

Table 5: Board power on a Raspberry Pi 5, 8 GB, measured at an independent laboratory on a Joulescope JS220, on flight 0104. The accelerator-fitted configurations were each measured 3 times and the row reports the first; the accelerator-removed configurations were measured 1 time each. Idle W is the mean of the 30 s idle interval recorded immediately before that run, under the same board configuration, and Incr. W is the diference: every row recomputes from its own two columns. The four idle baselines difer because the governor and core count change the idle draw as well as the loaded draw. ×RT is video seconds over wall seconds: below one means the pipeline did not keep up with its own footage. These runs are a separate execution of the system from the one profiled in Table 4 and scored in Section 5.3; see the text. For the accelerator-removed default configuration the source report lists 4.78 W in its table and 4.75289 W on the corresponding waveform; the latter is the one consistent with the reported energy and duration. The published table value is carried here unchanged. The discrepancy is the source report’s and is pending confirmation by its authors; were the waveform value confirmed, the incremental figure for that row would be 2.46 W.

## 5.6 Board power, measured independently

The runs in Table 4 report a power field that the software itself marks as not evidence, because no meter was attached to them. Table 5 is diferent: it was produced at the LaCASA laboratory of the University of Alabama in Huntsville, on their instrument, by people who did not write this system [7]. Every run directory from that bench still carries a null meter backend. None of these watts passed through our code.

A separate execution. On the same clip, with the same declared device card, the same declared execution provider, the same grounding backend and identical frame counts, the bench runs made 28 semantic calls and emitted 3 reports, where the runs profiled in Table 4 and scored in Section 5.3 made 12 and emitted 1. The manifests do not record whatever difers between them, so we can state the diference but not its cause. The consequence for the reader: the power and duration figures here are not a joint operating point with the detection results above, and no row of Table 5 should be attached to a recall or false-alarm figure from Section 5.3. Recording enough of the run configuration in the manifest to tell two executions apart is a defect this measurement exposed and is repaired in the current source.

Incremental power. Incremental power is 2.49 W with the accelerator fitted and 2.49 W without it. Idle board consumption is substantial, 2.83 W and 2.29 W respectively, so incremental processing power and total board power have to be reported separately, and the table does. The measured CPU-only bench configurations remained in the single-digit-watt range. We expected that, and it is the weakest claim in this section.

<table><tr><td>Flight</td><td>Code</td><td>Host</td><td>Warm</td><td>Obs</td><td>Staged</td><td>Found</td><td>FA/h</td></tr><tr><td>0101 control</td><td>post</td><td>macOS</td><td>yes</td><td>225</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>0102</td><td>pre</td><td>Linux</td><td>yes</td><td>288</td><td>1</td><td>1</td><td>0.0</td></tr><tr><td>0102</td><td>post</td><td>Linux</td><td>yes</td><td>235</td><td>1</td><td>1</td><td>0.0</td></tr><tr><td>0102</td><td>post</td><td>macOS</td><td>yes</td><td>232</td><td>1</td><td>1</td><td>0.0</td></tr><tr><td>0104</td><td>pre</td><td>Linux</td><td>yes</td><td>228</td><td>4</td><td>1</td><td>0.0</td></tr><tr><td>0104</td><td>post</td><td>Linux</td><td>no</td><td>33</td><td>4</td><td>0</td><td>0.0</td></tr><tr><td>0104</td><td>post</td><td>macOS</td><td>no</td><td>30</td><td>4</td><td>0</td><td>0.0</td></tr><tr><td>0106</td><td>post</td><td>macOS</td><td>yes</td><td>400</td><td>1</td><td>0</td><td>0.0</td></tr><tr><td>0107</td><td>post</td><td>macOS</td><td>no</td><td>111</td><td>1</td><td>0</td><td>0.0</td></tr></table>

Table 6: Every scored run of every staged flight. Code is before or after the speed-measurement correction described below. Obs is the number of observations the normality model accepted, against a warm-up floor of 120. A flight that never warms cannot fire a place-conditioned intent at all.

Governor and energy. With the accelerator fitted, taking two cores ofline and moving to the powersave governor cut mean power by 29.8% and lengthened the run by 51.7%, so the mission cost 6.55% more energy than at full speed. With the accelerator removed the same change lowered energy by 2.39%. The efect is configuration-specific and we do not generalise it. The caution generalises: a low-power result quoted as an average wattage, ours included, should be read beside the energy for the same work, because the configuration that looks better on one can be worse on the other.

Idle accelerator cost. The build measured here does not use the attached neural accelerator; the detector runs on the CPU. The unused accelerator adds about 0.5 W at idle; in the compared runs, the fitted configuration consumed 161 J more job energy. The lowest-energy configuration measured is the one with the accelerator physically removed.

Throughput against real time. At best it ran at 0.82× real time and in the low-power configuration at 0.54×. The independent bench is what confirmed it, but the evidence was in our own records: every run file carries a wall-clock duration alongside the clip length. The instrumentation does not compare them. Configured rates are expressed in media time: a 5 Hz detector is 5 calls per second of video whatever the clock says. That figure has to be set against the recorded wall-clock duration before anyone makes a real-time claim. A duty cycle measured in media time is not a claim about a live sensor, and Table 4 should be read with that in mind.

## 5.7 Replication, and a superseded result

Table 6 contains a result this paper reported and can no longer claim. On flight 0104 the pipeline named the silver sedan, and that is the detection Section 5.3 and Table 3 are built from. It was produced before a correction to how the tracker measures speed.

The correction: displacement is now the part of the motion on which both opposing edges of a box agree, so a box that grows on one side is no longer counted as travelling. It was made because a parked pickup measured 6.21 px/s against a 3.0 px/s ceiling and so never accumulated dwell. The normality model is taught only by tracks measuring above a speed floor, which is what keeps parked objects from teaching the model that judges them. Before the correction, flight 0104 fed it 228 observations; after it, 33. The diference was the parked vehicles’ own box jitter. With the jitter removed the flight holds too few genuine observations to warm the model, and the sedan is not reported. The same flight on a second host gives 30.

We report both runs because the pre-correction detection is the one the tick-level trace was taken from, and because the reason it disappeared is the finding. Flight 0102 is unafected: it carries real trafic, stays warm under both code versions and both hosts, and reports its staged handover every time.

Counting only the current code on one host, the pipeline names 1 of 7 staged events across four flights. That is the number this study supports.

## 5.8 The emission chain

Across the three synthetic runs, 34 emitted records were checked and all of them verify against the preceding hash. The unit test suite passes 148 of 148 with 0 failures.

## 6 Experiments not run

Nothing about the staged flights is held back. All five were flown on the same day and all five are scored in Table 6. The notebooks for flights 0106 and 0107 were written and hashed before either clip was run, and that ordering is recorded. Their outcomes are in the count whether or not they flatter the system.

Codec baselines. The comparison a reader will reasonably want, what H.264, H.265 or AV1 deliver at the same link budget and what a detect-and-track baseline sends, was not run alongside these results. A partial probe made afterwards is not reported here and does not change any number above: asked for the same 1 kbit/s budget on flight 0104, one AV1 encoder delivered 44× that rate, and it would only open at all with the rate cap removed, so it ran under diferent rate control from the other encoders and is not comparable to them. Until every encoder is measured under one rate-control setting there is no baseline table, and the ratios in Table 2 keep no external reference point.

Power on an airframe. No calibrated meter was attached to the runs in Table 4; their power field is flagged as not evidence and should be read that way. Section 5.6 reports board power measured independently on a single-board computer on a bench. It is not an airframe, its thermal environment is not flight-representative, and the accelerator the design assumes is fitted but unused.

Edge latency. The per-stage profile in Table 4 is host CPU wall time on a development machine: it characterises the schedule and the code on that host. The execution durations in Section 5.6 are embedded measurements, but they are whole-run durations on a diferent execution of the system. They are not per-stage or end-to-end cue latency on the configuration Table 4 profiles. No cue latency is reported on embedded hardware.

## 7 Threats to validity

The real-footage results rest on five flights, one of which staged nothing (Table 6). Across the four staged flights the pipeline matched one staged event of seven, and two of those four never warm their place model, so on those two it could not have reported whatever was in front of it. One detection and one clean flight support no rate, in either direction.

One semantic answer is counted many times. The evidence rule of Section 3 adds an increment on every detector tick, but the semantic term in that increment is the last answer the model gave, held until the gate next opens. On flight 0104 the semantic model ran 12 times while 1,443 increments accumulated, about 120 increments per distinct answer; on flight 0101, 7 against 2,010. The consequence is that the time at which the boundary is crossed depends on the detector rate as much as on the evidence: at half the detector rate the same footage and the same single semantic answer reach the boundary roughly twice as late. Boundaries derived for independent observations do not describe this process, which is why α and β are reported above as the place the test stops, and not as error rates. It also means a reduction in detector rate is a change in decision behaviour. It does not leave behaviour intact.

The staged scenes are a car park standing in for an overwatch task, and the normality model learns that specific scene, so a place-rarity score is not portable to another site. The structural thresholds are absolute pixel constants calibrated at one flying height, so dwell, cell size and the detector floor drift together as altitude changes; this is a known and unrepaired limitation. Ground truth on the real flights comes from an activity log written by the people who staged the events, and no independent annotation pass was made. The synthetic clips have exact truth but were produced by the same authors as the system under test.

A learned prior may not warm within one sortie. Two of the four staged flights never reached the warm-up floor of 120 observations: flight 0104 under the corrected tracker and flight 0107. Neither could fire a place-conditioned intent at any point, and both sent heartbeats throughout. The heartbeat is designed to separate a quiet scene from a dead radio; it does not encode this third state, in which the link is alive, the scene may not be quiet, and the system could not have reported. Each run records warm: false in its own manifest and puts nothing about it on the link. A per-flight prior needs trafic to learn from, and flight 0102, which has it, stays warm and reports under every configuration we ran. Whether the prior should persist across sorties is a design question this study does not settle, and lowering the floor would be fitting to these clips.

## 8 Reproducibility

No number in this paper is typed by hand. Each one is emitted from a result file by make numbers.py, which reads the JSON outputs the evidence pack hashes and writes the macro file this document includes, so a stale figure becomes a build error. The results are hashed into result digests.txt by a pack script that also records the host, the interpreter and the library versions as they actually ran. Each run records its own host, in the power block of its manifest, alongside the interpreter and library versions. Hosts are not uniform across this study and Table 6 gives the host for every scored run: the flight 0104 run traced in Table 3 was produced on Linux (aarch64, glibc 2.35), and the 14 September replication ran on macOS 26.5.1 on arm64. Both used Python 3.9.6. The synthetic clips regenerate from a fixed seed on every run and their hashes are recorded. The two grounding modes are distinguished in every result file by a grounding backend field and are never pooled.

## 9 Conclusion

A gated sensing pipeline is a sequence of exclusion decisions. Each one saves computation or communication, and each one creates a place where mission evidence can disappear. Tighten them and the link ratio improves, the semantic budget falls, and the things never examined grow. Loosen them and the reverse. A ratio reported without the misses describes one half of that mechanism.

On the evidence here the gate is doing real work and is not yet good enough. Under the corrected tracker it carried 1 of 7 staged events across four flights with no false alarm anywhere, and it was silent, correctly, for five minutes over an empty scene. It also let one event through to the semantic stage and stopped short, dropped a second on a normality score that the scene itself had taught it, and never saw a third because no track was formed. Those are three repairs in three stages, and the trace says which applies where.

We do not report a detection rate. Four staged flights and one detection is not a rate, two of those flights never warmed their normality model at all, and the compression ratio that would look best beside any of it is the one measured on the flight the system could not have reported on.

## Data and code availability

This manuscript is submitted with its source and numbers.tex, the generated macro file that carries every quoted value. The evidence pack that produced those values, meaning the result files, run logs, per-run manifests, the hash digest and make numbers.py, is held by the authors and is not distributed with the manuscript; numbers.tex gives the final figures and they do not carry the chain that produced them. The synthetic clips regenerate from the seed recorded in the manifest. The staged flight footage is not public and is not releasable. The runs in Table 6 dated 14 September are not yet in the hashed digest: the digest was generated before them and regenerating it moves every run identifier in this paper. Their manifests and scored outputs are in the run tree the pack reads, and the next pack regeneration will cover them.

## Funding and conflicts

Company-funded. No Government funding supported this work. The study was conducted alongside a United States Government Small Business Innovation Research proposal by the same authors; that is stated here.

## References

[1] D. G¨und¨uz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener, K. K. Wong and C.-B. Chae, “Beyond transmitting bits: context, semantics, and task-oriented communications,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 1, pp. 5–41, 2023. doi:10.1109/JSAC.2022.3223408

[2] D. Kang, J. Emmons, F. Abuzaid, P. Bailis and M. Zaharia, “NoScope: optimizing neural network queries over video at scale,” Proceedings of the VLDB Endowment, vol. 10, no. 11, pp. 1586–1597, 2017.

[3] W. P. M. H. Heemels, K. H. Johansson and P. Tabuada, “An introduction to event-triggered and self-triggered control,” Proc. 51st IEEE Conference on Decision and Control, Maui, HI, 2012, pp. 3270–3285.

[4] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger and I. Sutskever, “Learning transferable visual models from

natural language supervision,” Proc. 38th International Conference on Machine Learning, PMLR vol. 139, pp. 8748–8763, 2021.

[5] C. Staufer and W. E. L. Grimson, “Adaptive background mixture models for real-time tracking,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition, 1999.

[6] T. Bouwmans, “Traditional and recent approaches in background modeling for foreground detection: an overview,” Computer Science Review, vol. 11–12, pp. 31–66, 2014.

[7] V. Dzeletovi´c and A. Milenkovi´c, “Power profiling of Numberz AI application,” v1.0, LaCASA Laboratory, Department of Electrical and Computer Engineering, University of Alabama in Huntsville, 11 September 2026. Technical report; available from the authors.

[8] A. Wald, “Sequential tests of statistical hypotheses,” The Annals of Mathematical Statistics, vol. 16, no. 2, pp. 117–186, 1945. doi:10.1214/aoms/1177731118