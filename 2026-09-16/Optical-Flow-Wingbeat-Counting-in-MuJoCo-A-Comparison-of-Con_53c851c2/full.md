# Optical-Flow Wingbeat Counting in MuJoCo: A Comparison of Convolutional, Spiking, and Attention-Based Temporal Models

Zhang Nengbo

School of Aerospace Engineering, Engineering Campus Universiti Sains Malaysia, 14300 Nibong Tebal, Pulau Pinang, Malaysia zhangnb@student.usm.my

## Abstract

Visual monitoring of flapping-wing vehicles requires distinguishing individual wingbeats from motion strength and average frequency. This paper presents a controlled MuJoCo evaluation of wingbeat counting from signed optical flow observed by virtual cameras mounted on Crazyflie vehicles. Three flapping-wing models were recorded at optical distances of 1.5 and 3.0 m, producing 1,440 clips from 240 paired scene configurations with a scene-level 3:1 training–test split. A common spatial convolutional encoder was combined with a causal temporal convolutional network, a recurrent leaky integrate-and-fire spiking network, or causal self-attention. Each model predicted phase and activity, followed by the same directed-crossing event counter. The six existing convolutional models were retained, and all twelve new models were frozen before their test predictions were generated. Exact-count accuracies at 1.5 m were 96.67%, 95.00%, and 96.67%, respectively; at 3.0 m they were 94.44%, 92.22%, and 95.00%. All paired scene-bootstrap intervals for diferences in exact-count accuracy included zero. Seven far-distance spiking-model clips had correct totals despite event-timing mismatches, demonstrating why total-count and event-level measurements must be reported together. The results support the feasibility of causal optical-flow counting in the tested setting and identify boundary-sensitive errors. They do not establish an architecture ranking across repeated training, real-flight robustness, or hardware eficiency.

Keywords: wingbeat counting; optical flow; micro aerial vehicles; spiking neural networks; temporal convolution; causal attention; MuJoCo.

## 1 Introduction

Counting wingbeats from a camera stream provides a non-contact description of a flapping vehicle’s motion. The desired output can be a cumulative number of completed repetitions, the time of each repetition, or an average frequency. These outputs impose diferent requirements. A frequency estimate can describe a steady oscillation while leaving individual events unresolved. A correct total can also conceal compensating missed and extra detections, or events localized at the wrong times. An observation system intended to react to individual movements therefore needs a declared event definition and an evaluation that preserves event timing.

Visual wingbeat analysis and repetition counting have established precedents. High-speed optical-flow analysis has been used to estimate the wingbeat frequency of freely flying bumblebees [1]. Neural repetition counting has been studied with online convolutional processing [2], temporal self-similarity [3], attention-based temporal correlation [4], and motion-feature learning [5]. These results motivate an application-specific evaluation rather than a claim that optical flow or a particular neural architecture introduces repetition counting itself.

For micro aerial vehicles (MAVs), visually reading another vehicle’s movement is also relevant to motionmediated interaction. MoCom studied inter-MAV communication using event vision and spiking neural networks [6]. Wingbeat counting is a diferent perceptual task: the present output is an event sequence, not a decoded navigation message. The observer–target arrangement is useful because observation distance and target geometry can be controlled while keeping the sensor input and event labels explicit. Here, three independent observer–target pairs provide three target-specific tests; there is no cross-observer information fusion or communication protocol.

This study asks three bounded questions. Can a signed-optical-flow pipeline recover wingbeat events under a reproducible simulated observation protocol? How do convolutional, recurrent spiking, and attention-based temporal modules compare when the spatial encoder structure, training budget, and event counter are held fixed? Which disagreements remain when clip totals are correct? The study contributes an inspectable evaluation workflow, a paired comparison of three temporal designs, and an analysis of event-timing and window-boundary failures. It does not introduce a new general-purpose neural architecture or claim stateof-the-art repetition-counting performance. All reported observations come from completed experiments; untested extensions are identified in the discussion.

## 2 Related work

Wingbeat frequency and visual repetition counting. Santoyo et al. combined dense optical-flow contraction and expansion with frequency analysis and state estimation to measure bumblebee wingbeats [1]. Their work establishes a direct precedent for using optical flow to recover periodic wing motion. Live Repetition Counting used a convolutional network and temporal accumulation for incoming video [2]. RepNet learned a temporal self-similarity representation to estimate periods across action categories [3]. TransRAC introduced multiscale temporal correlation and fine-grained repetition annotations, including videos with interruptions and inconsistent actions [4]. Motion-feature learning further combined RGB and motion branches to address foreground motion and background changes [5]. The present experiment uses signed horizontal and vertical flow alone as its neural input. Its comparison is internal to one controlled dataset and does not reproduce these methods’ published benchmarks. In particular, published of-by-one or normalized counting errors should not be equated with the exact-count accuracy and unnormalized count MAE reported here.

Temporal model families. Causal temporal convolution is an established sequence-modeling approach [7]. Surrogate-gradient learning permits optimization of networks with hard spiking forward dynamics [8]. Selfattention [9] and attention with linear relative-position biases [10] provide another way to integrate temporal evidence. We use these components as alternative temporal modules following the same convolutional spatial encoder structure. Consequently, the spiking model is a hybrid analog-CNN/spiking model, and the attention model is a CNN/Transformer hybrid. The comparison does not assess fully spiking image encoders, pure vision Transformers, or neuromorphic hardware.

## 3 Observation task and reference events

## 3.1 Simulation and image acquisition

The observation scene was implemented in MuJoCo 3.10.0 [11]. Crazyflie geometry was based on the Crazyflie 2 model in MuJoCo Menagerie [12]. The three target assets were adapted from RLFlapping, flappy\_v2, and BIRD\_SIM [13, 14, 15]. Their configured nominal resting wingspans were 1.300, 0.40502, and 0.7504 m, and their baseline wing amplitudes were 0.45, 0.40, and 0.40 rad, respectively. These names identify imported geometries and joint structures; the experiments do not validate the flight performance or original aerodynamic assumptions of those projects.

Each Crazyflie carried one virtual forward-facing RGB camera and observed its corresponding target. Both body roots were nominally held at a height of 1.3 m. The camera optical center was placed either 1.5 or 3.0 m from the target root. Each image contained the assigned target at a resolution of 128 × 96 pixels, with a 45<sup>◦</sup> vertical field of view and a 60 Hz acquisition rate. Saved images were not digitally cropped or resized. The simulator ran at 2,400 Hz, giving 40 physics steps per image. Shadow rendering was disabled.

An ideal inverse-dynamics controller applied generalized forces to stabilize the bodies and track prescribed sinusoidal wing motions. For the simulated generalized position error � and velocity reference ${ \dot { q } } _ { \mathrm { r e f } }$ , the inverse-dynamics acceleration request was $\ddot { q } _ { \mathrm { r e f } } + 1 6 0 0 e + 8 0 ( \dot { q } _ { \mathrm { r e f } } - \dot { q } )$ . The resulting force was applied before each MuJoCo integration step. This establishes a controlled visual-motion experiment; it is not a demonstration that aerodynamic wing forces independently sustain flight. Observers remained nominally stationary during acquisition, and pose stabilization used simulator state separately from the visual counting pipeline.

## 3.2 Paired scenes and split

The dataset contained 240 paired scene configurations. Each configuration produced three target-camera streams at each distance, giving six clips per configuration and 1,440 clips overall. Each clip contained 180 frames with timestamps $t _ { i } = i / 6 0 , i = 0 , . . . , 1 7 9$ , after a separate 0.8 s simulation settling period. The complete collection contained 259,200 RGB frames. The scene-generation seed was 20261001.

The first designated 180 configurations were assigned to training and 60 to testing. All three targets and both distances of a configuration remained in the same split. Thus, each target–distance condition contained 180 training and 60 test clips, while each distance pooled over targets contained 540 training and 180 test clips. This was a 3:1 split by complete paired scene, not by overlapping frames or windows.

Moving-scene frequencies were drawn in the range 2.5–7.5 Hz. A scheduled subset used permutations of 3, 5, and 7 Hz; each clip retained a constant commanded frequency. Initial phase, wing amplitude, illumination, background and floor colors, and target appearance varied across scenes. Amplitudes were scaled by factors sampled from [0.75, 1.2] relative to the configured target amplitude. Paired distances shared these choices. Ten percent of the scenes in each split were stationary controls: 18 training and six test scenes per distance, yielding 18 static test clips when all three targets were pooled. These controls replaced the corresponding moving commands. They were retained in all primary metrics.

## 3.3 Reference phase and one-event-per-cycle definition

Reference labels came from the actual primary-wing joint position and velocity at image times. For a moving target, let $q _ { i }$ be the signed primary-wing displacement relative to its neutral position, ${ \dot { q } } _ { i }$ its signed velocity, � the configured amplitude, and � the commanded frequency. The teacher phase was

$$
\phi _ { i } = \mathrm { u n w r a p } \left[ \mathrm { a t a n 2 } \left( \frac { q _ { i } } { A } , \frac { \dot { q } _ { i } } { 2 \pi f A } \right) \right] .\tag{1}
$$

Within the learning and evaluation pipeline, commanded frequency and amplitude were used for teacher annotation only; neither was supplied to optical-flow estimation or neural inference. They also defined the simulated motion commands. Stationary clips received a constant reference phase and zero events. Saved labels were checked against the actual joint’s negative-to-positive mid-position crossings.

One wingbeat event was defined as the primary wing crossing its neutral position in the positive direction, equivalently crossing an increasing 2� phase boundary. Synchronously moving left and right wings did not count as two separate events. Event times were linearly interpolated between adjacent reference phase samples. This convention counts observed cycle landmarks, and is distinct from taking the floor of total phase advance or multiplying a commanded frequency by clip duration.

The first 30 image frames were used for history preparation. All architectures were evaluated on the same interval

$$
{ \cal { I } } = \left( t _ { 3 0 } , t _ { 1 7 9 } \right) = \left( { 0 . 5 , 2 . 9 8 3 \overline { { { 3 } } } } \right) { \bf { s } } , \qquad | { \cal { I } } | = { 1 4 9 / 6 0 } { \bf { s } } .\tag{2}
$$

An event exactly at the interval start was excluded. No frame after the clip end was available to confirm a final candidate event.

## a Three independent observer–target pairs

Separately acquired at 1.5 m and 3.0 m; nominal body-root height: 1.3 m  
![](images/e6de055971b3df815778b238ec27ea83243b228bc9afae21f685e6097d02c118.jpg)  
b Identical counting interface; independently trained models

Choose one temporal module per model; each target and distance is trained separately  
![](images/5323d5d6b5994b18b215f78021ba4517e6c9562bfd96de7cceb890e80ff8cef4.jpg)  
Figure 1: Observation and counting information flow. Three independent camera–target pairs are evaluated at two separately acquired distances. Each model reads signed optical flow from its own RGB stream. The three temporal modules are alternatives, not an ensemble or a multi-observer fusion mechanism. They use the same spatial encoder structure with independently trained weights. Simulator phase is reserved for training and reference-event evaluation; it does not enter the inference path. The counter records an interpolated crossing time only after causal confirmation. The drawing is schematic and not to scale.

## 4 Optical-flow phase prediction and event counting

## 4.1 Inputs and shared spatial encoder

Successive RGB images were converted to grayscale and processed by Farnebäck dense optical flow [16]. The fixed OpenCV parameters were pyramid scale 0.5, three levels, window size 15, three iterations, polynomial neighborhood 5, polynomial standard deviation 1.2, and flags 0. The resulting input $F _ { i } \in \mathbb { R } ^ { 9 6 \times 1 2 8 \times \bar { 2 } }$ contained signed horizontal and vertical displacements in pixels per frame. The first flow field in each clip was set to zero. Neither simulator segmentation nor label-dependent target crops were supplied to the network.

The neural pipeline is summarized in Fig. 1. Flow values were transformed component-wise as

$$
\widetilde { F } _ { i } = \frac { 1 } { 3 } \mathrm { a s i n h } \left( \frac { \mathrm { c l i p } ( F _ { i } , - 8 , 8 ) } { 0 . 1 } \right) .\tag{3}
$$

Three $5 \times 5$ convolutions with stride 2 and padding 2 produced 12, 24, and 32 channels. Each stage used four-group normalization and a GELU activation. Adaptive average pooling to $3 \times 4$ spatial bins and a linear projection yielded a 62-dimensional vector. Two additional features, log(1 + mean $\| F _ { i } \| _ { 2 } )$ and log(1 + max $\| F _ { i } \| _ { 2 } )$ , preserved raw motion-magnitude information for stationary decisions. Concatenation gave a 64-dimensional frame representation. The spatial encoder contained 51,074 trainable parameters. Its structure and initial weights were matched within a target across model families and distances; subsequent training optimized independent weights for every model.

## 4.2 Alternative temporal modules

All modules predicted an unnormalized two-dimensional phase vector $p _ { i } = ( p _ { i } ^ { c } , p _ { i } ^ { s } )$ and an activity logit $\ell _ { i }$ The phase angle was $\hat { \phi } _ { i } = \mathrm { a t a n } 2 ( p _ { i } ^ { s } , p _ { i } ^ { c } )$ and the activity probability was $a _ { i } = \sigma ( \ell _ { i } )$ .

Table 1: Compared models. Parameter totals include the common spatial encoder. Approximately matched parameter totals and equal optimizer steps do not imply matched temporal context, compute, or energy.
<table><tr><td>Model</td><td>Parameters</td><td>Temporal context</td></tr><tr><td>CNN-TCN</td><td>117,317</td><td>Fixed 31-frame causal receptive field.</td></tr><tr><td>CNN-LIF</td><td>116,077</td><td>Recurrent state over preceding frames since reset.</td></tr><tr><td>CNN-Transformer</td><td>118,341</td><td>Causal attention over the supplied preceding frames.</td></tr></table>

Convolutional temporal processing (CNN–TCN). Four residual temporal blocks used kernel size 3, 64 channels, and dilations 1, 2, 4, and 8. Left padding made every convolution causal. Each block applied GELU, dropout 0.05, a pointwise channel mixer, and a residual connection. A final pointwise convolution produced the three outputs. The receptive field was $1 + 2 ( 1 + 2 + 4 + 8 ) = 3 1$ frames.

Recurrent spiking temporal processing (CNN–LIF). The frame representation passed through layer normalization and a linear projection to two recurrent leaky integrate-and-fire (LIF) layers, each containing 136 neurons. For layer �, the per-frame update was

$$
u _ { i } ^ { l , - } = \beta u _ { i - 1 } ^ { l , + } + W _ { l } x _ { i } ^ { l } + R _ { l } s _ { i - 1 } ^ { l } + b _ { l } ,\tag{4}
$$

$$
\begin{array} { r } { s _ { i } ^ { l } = \mathbf { 1 } [ u _ { i } ^ { l , - } \geq \vartheta ] , \qquad u _ { i } ^ { l , + } = u _ { i } ^ { l , - } - \vartheta \ : \mathrm { s g } ( s _ { i } ^ { l } ) , } \end{array}\tag{5}
$$

where $x _ { i } ^ { 2 } = s _ { i } ^ { 1 } , \beta = 0 . 8 , \vartheta = 1$ , and sg denotes a stopped reset gradient. The first-layer input was the normalized 64-dimensional encoder representation. Training used the surrogate derivative $( 1 + 5 | u _ { i } ^ { l , - } - \vartheta | ) ^ { - 2 } \left[ 8 \right]$ . The readout operated only on a second-layer spike trace $z _ { i } = 0 . 5 z _ { i - 1 } + 0 . 5 s _ { i } ^ { 2 }$ ; no direct analog input or membrane bypass fed the output layer. Each image frame corresponded to one spiking step, with no Poisson encoding or hidden microsteps. States were zeroed for independent windows and clips. This is a dense-GPU implementation of a hybrid network, without a measured energy advantage.

Causal attention (CNN–Transformer). Two pre-normalized attention blocks used width 64, four heads, a feed-forward width of 128, GELU, and dropout 0.05. Future attention positions were masked. ALiBi relative-position penalties were added to attention scores [10], with head slopes 0.25, 0.0625, 0.015625, and 0.00390625. No absolute timestamp or externally supplied phase was embedded in the features. A final layer normalization and linear readout produced phase and activity. Attention could use all preceding frames in the supplied window or clip.

## 4.3 Learning objective

Let $m \in \{ 0 , 1 \}$ indicate whether a training clip contained commanded motion and $y _ { i } = m ( \cos \phi _ { i } , \sin \phi _ { i } )$ . The objective, evaluated after the first 30 frames of a sampled window, was

$$
\mathcal { L } = \mathbf { M S E } ( p , y ) + 0 . 2 5 \mathbf { M S E } ( \Delta p , \Delta y ) + 0 . 1 5 \mathbf { B C E W i t h L o g i t s } ( \ell , m ) + 0 . 0 2 \mathbf { M S E } ( \| p \| _ { 2 } , m ) .\tag{6}
$$

The finite diferences retained direction of phase progression. The stationary phase-vector target was zero, so its arbitrary constant teacher phase did not contribute a motion target. The loss components were kept identical across model families. Their individual benefit was not established by an ablation experiment in this study.

## 4.4 Causal event counter

The same stateful counter processed every model’s outputs. A sample was active when $a _ { i } > 0 . 5$ . An active phase observation required $\| p _ { i } \| _ { 2 } \ge 0 . 2 5$ and finite values. Finite inactive predictions were available stationary decisions even when the phase vector was zero. Unavailable observations, inactivity, or timestamp gaps exceeding 1.5/60 s cleared phase tracking while preserving the accumulated count.

Within an active segment, wrapped angular diferences were integrated to track increasing phase. A candidate boundary 2�� became armed after the estimated phase reached its negative hysteresis side, 2�� − 0.1.

A positive zero crossing created a candidate with an interpolated crossing time. It was confirmed only when phase reached $2 \pi k + 0 . 1$ . Confirmed events had to be separated by at least 0.08 s, and the same boundary could not be emitted twice. The cumulative count increased at confirmation time, whereas the saved crossing timestamp was used for event matching. The latter is retrospective interpolation of an already confirmed event, not permission to display a count before it became available.

The counter never filled missing observations by extrapolating commanded frequency. Likewise, a candidate near the end of the clip that had not reached positive hysteresis was not counted. At the evaluation start, the counter initialized from the current phase rather than importing a previously armed event from the warm-up period. This deliberate boundary convention can lose a reference event very close to either endpoint and is retained in the reported errors.

## 5 Training and evaluation protocol

## 5.1 Training budget and retained baseline

Each target and distance had independently trained models, giving 18 models across the three temporal families. Every model used 60 epochs, 24 optimizer steps per epoch, batch size 6, and 96-frame windows. An epoch comprised 24 minibatches sampled with replacement, rather than a full traversal of the training partition; the total was 1,440 updates and 8,640 sampled windows per model. AdamW used an initial learning rate of $1 0 ^ { - 3 }$ weight decay $1 0 ^ { - 4 }$ , cosine decay to $5 \times 1 0 ^ { - 5 }$ , and gradient-norm clipping at 5. Each sampled window received one positive flow-magnitude gain drawn uniformly from [0.9, 1.1]. The full 180-clip training partition was eligible for sampling. There was no validation-based model selection, early stopping, or architecture-specific hyperparameter search; epoch 60 was retained.

Seeds were 2026091603, 2026091703, and 2026091803 for RLFlapping, flappy\_v2, and BIRD\_SIM, respectively, with the same target seed used across distances and architectures. Initialization hashes verified identical starting spatial encoders. Recorded sampling hashes verified identical clip/window selection schedules. Gain distributions were the same, but gain values were not guaranteed to be identical because the architectures consumed the shared CUDA random-number stream diferently. TF32 was disabled and deterministic settings were requested with warnings rather than hard failures. Adaptive-pooling and attention backward operations can retain numerical nondeterminism; bitwise retraining reproducibility is not claimed.

The six CNN–TCN models and their evaluation reports had been completed in the preceding optical-flow experiment. They were audited and retained byte-for-byte, rather than retrained or reselected. Twelve CNN–LIF and CNN–Transformer models were then trained under a written comparison protocol. All twelve checkpoints were frozen, and hashes of all 18 checkpoints were checked, before any new test prediction was generated. No test-based threshold adjustment or checkpoint selection followed these predictions. Training used Python 3.10.20, PyTorch 2.11.0+cu128, OpenCV 4.13.0, and two NVIDIA GeForce RTX 5060 Ti GPUs.

The test scenes had previously been used during development of the optical-flow approach. Consequently, this is a reused fixed-split development comparison, not evaluation on a newly collected external holdout. Each model has one training realization. Near- and far-distance models were trained separately, so the comparison does not demonstrate transfer of one trained model between distances.

## 5.2 Count and event metrics

For � clips, let $N _ { j }$ and $\hat { N } _ { j }$ denote the reference and predicted event counts within Eq. (2). Exact-count accuracy and unnormalized count mean absolute error were

$$
\mathrm { A c c } _ { \mathrm { e x a c t } } = \frac { 1 } { M } \sum _ { j } { \bf 1 } [ \hat { N } _ { j } = N _ { j } ] , \qquad \mathrm { M A E } _ { \mathrm { c o u n t } } = \frac { 1 } { M } \sum _ { j } | \hat { N } _ { j } - N _ { j } | .\tag{7}
$$

All clips remained in the denominator, including stationary clips and clips with unavailable observations. The unit of count MAE is events per clip.

Table 2: Main results, pooling three target types. Each row contains 180 clips from 60 paired scenes, including 18 static clips. Accuracy and its interval are percentages. Intervals use paired scene-cluster bootstrap resampling; F1 uses micro event totals.
<table><tr><td>Distance</td><td>Model</td><td>Exact</td><td>Accuracy</td><td>95% interval</td><td>Count MAE</td><td>Event F1</td></tr><tr><td>1.5</td><td>CNN-TCN</td><td>174/180</td><td>96.67</td><td>[93.89, 98.89]</td><td>0.0333</td><td>0.998470</td></tr><tr><td>1.5</td><td>CNN-LIF</td><td>171/180</td><td>95.00</td><td>[91.67, 97.78]</td><td>0.0500</td><td>0.997705</td></tr><tr><td>1.5</td><td>CNN-Transformer</td><td>174/180</td><td>96.67</td><td>[93.89, 98.89]</td><td>0.0333</td><td>0.998470</td></tr><tr><td>3.0</td><td>CNN-TCN</td><td>170/180</td><td>94.44</td><td>[91.11, 97.22]</td><td>0.0556</td><td>0.997449</td></tr><tr><td>3.0</td><td>CNN-LIF</td><td>166/180</td><td>92.22</td><td>[87.78, 96.11]</td><td>0.0778</td><td>0.992861</td></tr><tr><td>3.0</td><td>CNN-Transformer</td><td>171/180</td><td>95.00</td><td>[91.67, 97.78]</td><td>0.0500</td><td>0.997706</td></tr></table>

Predicted and reference event timestamps were matched one-to-one within a tolerance of 2/60 s (two camera frames). Matching first maximized the number of pairs and then minimized total timing error among tied solutions. Matched events were true positives (TP); unmatched predictions were false positives (FP); unmatched references were false negatives (FN). Precision, recall, and F1 were computed from pooled event totals, with $\mathrm { F } 1 = 2 \mathrm { T P } / ( 2 \mathrm { T P } + \mathrm { F P } + \mathrm { F N } )$ . We did not average per-clip F1, which would allow zero-event static clips to inflate an event metric. FP and FN describe unmatched events under this tolerance and can arise from timing displacement even when clip totals agree.

## 5.3 Paired uncertainty and implementation checks

The 180 test clips per distance came from 60 scene configurations with three targets each. We therefore resampled paired scene clusters rather than treating all 180 clips as independent. Ten thousand bootstrap replicates used seed 2026091609, retaining all three targets and all architecture/distance observations of a selected scene together. The same resampled scene indices were used for model contrasts. Equal-tailed 95% intervals were the 2.5th and 97.5th percentiles. These are individual, unadjusted intervals conditional on the fixed trained weights; they do not quantify variation across retraining or support simultaneous claims across all contrasts.

Separate synthetic checks verified that future inputs did not change earlier outputs, that the LIF forward path emitted binary spikes with reset, and that chunked LIF inference with carried state matched full-sequence inference. They also checked encoder initialization and frozen-source integrity. Each model received an additional 180-frame all-zero optical-flow control. These checks establish specific implementation properties, not robustness to untested camera motion or real-world backgrounds.

## 6 Results

## 6.1 Exact counts and event-level performance

At 1.5 m, CNN–TCN and CNN–Transformer each counted 174 of 180 clips exactly, while CNN–LIF counted 171 exactly (Table 2). At 3.0 m, the corresponding counts were 170, 171, and 166 for CNN–TCN, CNN–Transformer, and CNN–LIF. Thus, the far-distance advantage of CNN–Transformer over CNN–TCN was one correctly counted clip, or 0.56 percentage points. Count MAE ranged from 0.0333 to 0.0778 events per clip. These figures describe the configured models within the tested conditions.

There were 1,964 reference events per distance, identical by construction across the paired observations (Table 3). Near-distance CNN–TCN and CNN–Transformer each produced 1,958 matched events with no unmatched predictions. Far-distance CNN–LIF produced 1,947 matched events, 11 unmatched predictions, and 17 unmatched references, yielding F1 0.992861. Its exact-count accuracy remained 92.22%, illustrating that a high fraction of correct totals does not summarize all event-timing errors.

Table 3: Reference and predicted event totals and one-to-one matching. P and R denote micro precision and recall in percent. An unmatched event may reflect a timing error, a missed cycle, or an extra detected cycle.
<table><tr><td>Distance</td><td>Model</td><td>Ref.</td><td>Pred.</td><td>TP</td><td>FP</td><td>FN</td><td>P</td><td>R</td></tr><tr><td>1.5</td><td>CNN-TCN</td><td>1964</td><td>1958</td><td>1958</td><td>0</td><td>6</td><td>100.000</td><td>99.695</td></tr><tr><td>1.5</td><td>CNN-LIF</td><td>1964</td><td>1957</td><td>1956</td><td>1</td><td>8</td><td>99.949</td><td>99.593</td></tr><tr><td>1.5</td><td>CNN-Transformer</td><td>1964</td><td>1958</td><td>1958</td><td>0</td><td>6</td><td>100.000</td><td>99.695</td></tr><tr><td>3.0</td><td>CNN-TCN</td><td>1964</td><td>1956</td><td>1955</td><td>1</td><td>9</td><td>99.949</td><td>99.542</td></tr><tr><td>3.0</td><td>CNN-LIF</td><td>1964</td><td>1958</td><td>1947</td><td>11</td><td>17</td><td>99.438</td><td>99.134</td></tr><tr><td>3.0</td><td>CNN-Transformer</td><td>1964</td><td>1959</td><td>1957</td><td>2</td><td>7</td><td>99.898</td><td>99.644</td></tr></table>

Table 4: Target-specific results. Every row includes 60 clips, of which six are static. Distance is in meters, accuracy in percent, and count MAE in events per clip.
<table><tr><td>Distance</td><td>Model</td><td>Target</td><td>Exact</td><td>Accuracy</td><td>Count MAE</td><td>Event F1</td></tr><tr><td>1.5</td><td>CNN-TCN</td><td>RLFlapping</td><td>58/60</td><td>96.67</td><td>0.033</td><td>0.99840</td></tr><tr><td>1.5</td><td>CNN-TCN</td><td>flappy_v2</td><td>60/60</td><td>100.00</td><td>0.000</td><td>1.00000</td></tr><tr><td>1.5</td><td>CNN-TCN</td><td>BIRD_SIM</td><td>56/60</td><td>93.33</td><td>0.067</td><td>0.99707</td></tr><tr><td>1.5</td><td>CNN-LIF</td><td>RLFlapping</td><td>56/60</td><td>93.33</td><td>0.067</td><td>0.99680</td></tr><tr><td>1.5</td><td>CNN-LIF</td><td>flappy_v2</td><td>58/60</td><td>96.67</td><td>0.033</td><td>0.99847</td></tr><tr><td>1.5</td><td>CNN-LIF</td><td>BIRD_SIM</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99780</td></tr><tr><td>1.5</td><td>CNN-Transformer</td><td>RLFlapping</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99760</td></tr><tr><td>1.5</td><td>CNN-Transformer</td><td>flappy_v2</td><td>60/60</td><td>100.00</td><td>0.000</td><td>1.00000</td></tr><tr><td>1.5</td><td>CNN-Transformer</td><td>BIRD_SIM</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99780</td></tr><tr><td>3.0</td><td>CNN-TCN</td><td>RLFlapping</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99760</td></tr><tr><td>3.0</td><td>CNN-TCN</td><td>flappy_v2</td><td>58/60</td><td>96.67</td><td>0.033</td><td>0.99847</td></tr><tr><td>3.0</td><td>CNN-TCN</td><td>BIRD_SIM</td><td>55/60</td><td>91.67</td><td>0.083</td><td>0.99633</td></tr><tr><td>3.0</td><td>CNN-LIF</td><td>RLFlapping</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99760</td></tr><tr><td>3.0</td><td>CNN-LIF</td><td>flappy_v2</td><td>54/60</td><td>90.00</td><td>0.100</td><td>0.98471</td></tr><tr><td>3.0</td><td>CNN-LIF</td><td>BIRD_SIM</td><td>55/60</td><td>91.67</td><td>0.083</td><td>0.99633</td></tr><tr><td>3.0</td><td>CNN-Transformer</td><td>RLFlapping</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99760</td></tr><tr><td>3.0</td><td>CNN-Transformer</td><td>flappy_v2</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99770</td></tr><tr><td>3.0</td><td>CNN-Transformer</td><td>BIRD_SIM</td><td>57/60</td><td>95.00</td><td>0.050</td><td>0.99780</td></tr></table>

## 6.2 Target-specific behavior

Table 4 retains all target-specific results. Near-distance flappy\_v2 was counted exactly in all 60 clips by CNN–TCN and CNN–Transformer. This finite result is specific to that model, geometry, and test partition. At the far distance, the largest event-F1 reduction occurred for CNN–LIF on flappy\_v2: 54 of 60 totals were exact, but event F1 was 0.98471. The other two far-distance CNN–LIF target conditions had F1 above 0.995. The observed diference therefore did not occur uniformly across target types. Because the assets have diferent physical wingspans, equal observation distances do not equalize their projected image sizes; target-specific diferences cannot be attributed to temporal processing alone.

## 6.3 Paired diferences

All 95% intervals for pairwise diferences in exact-count accuracy included zero (Table 5). For example, the far-distance CNN–Transformer minus CNN–TCN diference was +0.56 percentage points with interval [−1.67, +2.78]. The point estimates therefore do not establish a stable exact-count ranking, and an interval spanning zero is not an equivalence test.

The far-distance event-F1 diferences between CNN–LIF and the other two models had individual intervals excluding zero: CNN–LIF minus CNN–TCN was −0.004588 with interval [−0.008936, −0.001059], and CNN–Transformer minus CNN–LIF was +0.004845 with interval [+0.001285, +0.009106]. These are conditional, unadjusted contrasts over the same test scenes. They should not be generalized to all training seeds or used as a claim that spiking temporal processing is intrinsically inferior.

Table 5: Paired model diferences, written as first minus second, with individual 95% scene-bootstrap intervals. Accuracy diferences are percentage points; F1 diferences use the 0–1 scale. The intervals do not include training-seed variability and are not corrected for multiple comparisons.
<table><tr><td>Distance Contrast</td><td></td><td>∆ exact accuracy [95% interval]</td><td>∆ event F1 [95% interval]</td></tr><tr><td>1.5</td><td> $\mathrm { C N N - L I F - C N N - T C N }$ </td><td> $- 1 . 6 7 \ [ - 4 . 4 4 , + 1 . 1 1 ]$ </td><td> $- 0 . 0 0 0 7 6 6 [ - 0 . 0 0 2 0 9 8 , + 0 . 0 0 0 5 2 1 ]$ </td></tr><tr><td>1.5</td><td> $\mathrm { C N N - T r a n s f o r m e r - C N N - T C N }$ </td><td> $+ 0 . 0 0 \left[ - 1 . 6 7 , + 1 . 6 7 \right]$ </td><td> $ + 0 . 0 0 0 0 0 0 [ - 0 . 0 0 0 7 5 5 , + 0 . 0 0 0 7 5 3 ]$ </td></tr><tr><td>1.5</td><td> $\mathrm { C N N - T r a n s f o r m e r - C N N - L I F }$ </td><td> $+ 1 . 6 7 \ [ - 0 . 5 6 , + 4 . 4 4 ]$ </td><td> $+ 0 . 0 0 0 7 6 6 [ - 0 . 0 0 0 2 6 4 , + 0 . 0 0 1 9 3 8 ]$ </td></tr><tr><td>3.0</td><td> $\mathrm { C N N - L I F - C N N - T C N }$ </td><td> $- 2 . 2 2 \ [ - 6 . 6 7 , + 1 . 6 7 ]$ </td><td> $- 0 . 0 0 4 5 8 8 \ [ - 0 . 0 0 8 9 3 6 , - 0 . 0 0 1 0 5 9 ]$ </td></tr><tr><td>3.0</td><td> $\mathrm { C N N - T r a n s f o r m e r - C N N - T C N }$ </td><td> $+ 0 . 5 6 \ [ - 1 . 6 7 , + 2 . 7 8 ]$ </td><td> $+ 0 . 0 0 0 2 5 7 \ [ - 0 . 0 0 0 8 1 0 , + 0 . 0 0 1 3 7 1 ]$ </td></tr><tr><td>3.0</td><td> $\mathrm { C N N - T r a n s f o r m e r - C N N - L I F }$ </td><td> $+ 2 . 7 8 \ [ - 1 . 1 1 , + 7 . 2 2 ]$ </td><td> $+ 0 . 0 0 4 8 4 5 \ [ + 0 . 0 0 1 2 8 5 , + 0 . 0 0 9 1 0 6 ]$ </td></tr></table>

Table 6: Controls and availability. Static errors are incorrectly counted static clips out of 18; zero-flow failures are models producing a nonzero count out of three. Unavailable records were not excluded. Moving accuracy uses only 162 moving clips and is expressed in percent.
<table><tr><td>Distance</td><td>Model</td><td>Static errors</td><td>Zero-flow failures</td><td>Unavailable</td><td>Moving accuracy</td></tr><tr><td>1.5</td><td>CNN-TCN</td><td>0/18</td><td>0/3</td><td>1</td><td>96.30</td></tr><tr><td>1.5</td><td>CNN-LIF</td><td>0/18</td><td>0/3</td><td>0</td><td>94.44</td></tr><tr><td>1.5</td><td>CNN-Transformer</td><td>0/18</td><td>0/3</td><td>0</td><td>96.30</td></tr><tr><td>3.0</td><td>CNN-TCN</td><td>0/18</td><td>0/3</td><td>0</td><td>93.83</td></tr><tr><td>3.0</td><td>CNN-LIF</td><td>0/18</td><td>0/3</td><td>1</td><td>91.36</td></tr><tr><td>3.0</td><td>CNN-Transformer</td><td>0/18</td><td>0/3</td><td>1</td><td>94.44</td></tr></table>

## 6.4 Stationary controls and observation availability

No architecture produced a false event in the 18 stationary test clips at either distance. All 18 additional zero-flow controls also produced zero counts (Table 6). Three architecture–clip records had at least one unavailable observation: one near CNN–TCN record, one far CNN–LIF record, and one far CNN–Transformer record. They were retained in every primary metric. Because static clips are easier to count as zero, Table 6 also reports accuracy on the 162 moving clips per distance.

## 6.5 Post-hoc boundary and timing analysis

Saved event lists were inspected without changing checkpoints, predictions, or counting thresholds. Every unmatched event for CNN–TCN, CNN–Transformer, and near-distance CNN–LIF lay within one camera frame of an evaluation endpoint (Table 7). This concentration is consistent with the counter’s requirement to arm after the interval begins and to confirm before the clip ends. The preceding CNN–TCN inspection distinguished initial unarmed crossings, final unconfirmed candidates, and final predicted phases that had not yet crossed. These are part of the current protocol’s measured error, rather than observations removed from evaluation.

Far-distance CNN–LIF additionally had seven unmatched reference events and seven unmatched predicted events outside the endpoint neighborhoods. Seven far-distance CNN–LIF clips had equal predicted and reference totals despite nonzero FP and FN. Correct totals in those cases did not imply accurate event localization. The stored timing mismatches, rather than total-count accuracy alone, identified this failure mode. This post-hoc description does not isolate its architectural cause and was not used for test-driven revisions.

## 7 Discussion and limitations

The main empirical finding is that three modest temporal models can support causal wingbeat counting from signed optical flow in this controlled task, while exact totals and correctly timed events remain distinct outcomes. A second finding is the strong influence of evaluation endpoints on most observed errors. With a count interval of only 149/60 s, losing one event near a boundary makes an otherwise accurate clip fail exact-count evaluation. Changing the boundary protocol could change the reported percentage. A future protocol should specify whether state may carry across windows, whether delayed confirmation after a reporting boundary is allowed, and how confirmation delay is scored, before evaluating new models.

Table 7: Post-hoc error locations. Boundary FN and FP are counts within 1/60 s of either endpoint divided by all FN or FP in that condition. A zero denominator means there were no errors of that type. The last column counts clips whose totals are exact but whose event matching has an error.
<table><tr><td>Distance</td><td>Model</td><td>Boundary FN / all FN</td><td>Boundary FP / all FP Exact but mismatched</td></tr><tr><td>1.5</td><td>CNN-TCN</td><td>6/6</td><td>0/0</td></tr><tr><td>1.5</td><td>CNN-LIF</td><td>8/8</td><td>1/1 0</td></tr><tr><td>1.5</td><td>CNN-Transformer</td><td>6/6</td><td>0/0 0 0</td></tr><tr><td>3.0</td><td>CNN-TCN</td><td>9/9</td><td>1/1 7</td></tr><tr><td>3.0</td><td>CNN-LIF</td><td>10/17</td><td>4/11</td></tr><tr><td>3.0</td><td>CNN-Transformer</td><td>7/7</td><td>2/2</td></tr></table>

The comparison controls several factors but does not isolate an architecture efect. Spatial encoder structure, initial encoder weights, scene split, sampled windows, loss, optimizer schedule, and counter thresholds were matched. Parameter totals were close. However, temporal context difered, random augmentation draws were not identical, and each architecture used one fixed hyperparameter configuration and one training realization per condition. The CNN baseline was reused after prior evaluation. These choices support a practical development comparison, not a universal ranking or a test of optimally tuned representatives of each model family.

The simulation also limits the application claim. Frequencies were constant within clips, cameras were nominally stationary, and visual variation consisted mainly of controlled appearance and illumination changes. There were no systematic tests of observer ego motion, continuous frequency change, long-duration drift, occlusion, motion blur, dropped frames, or realistic camera noise. The near and far models were trained separately; no target type, distance, or background family was withheld as an unseen domain. Ideal generalizedforce stabilization does not establish aerodynamic free-flight control. Three separate observer streams likewise do not establish cooperative perception or communication.

Causal information flow is necessary for an online implementation, but it is not a measurement of real-time deployment. The current evaluation processed saved clips, and no end-to-end camera-to-event latency, sustained onboard throughput, or hardware energy was measured. The spiking path was implemented with dense GPU operations after an analog CNN, so an energy-eficiency claim cannot be inferred from its binary spikes. A deployment study would have to include optical-flow computation, spatial encoding, temporal state handling, and event confirmation latency in the same measurement.

Finally, this work contains an internal architecture comparison, not a reproduced comparison against established repetition-counting methods or a classical peak/phase-tracking baseline. The relevant literature supplies prior approaches and context, not directly comparable scores on this dataset. A stronger follow-up would lock a new scene-disjoint holdout, repeat training across seeds, compare appropriate causal baselines, and test real video of a flapping mechanism before making broader robotic claims. The present version preserves its completed measurements and explicit failure cases as a starting point for those tests.

## 8 Conclusion

A controlled MuJoCo workflow was developed to count individual wingbeat events from signed optical flow using alternative convolutional, recurrent spiking, and attention-based temporal modules. Near-distance exact-count accuracy ranged from 95.00% to 96.67%, and far-distance accuracy from 92.22% to 95.00%, under the same fixed event counter. Paired intervals did not establish an exact-count architecture ranking. Event-level analysis revealed both endpoint-sensitive errors and seven far-distance spiking-model clips with correct totals but inaccurate event matching. These results support an inspectable simulation study of wingbeat counting and motivate evaluating event timing, confirmation, and cumulative counts together when extending the task to moving cameras and real platforms.

## Data, code, and model availability

The accompanying source submission includes the experiment protocol, aggregate statistics, and per-clip count and event-matching records as ancillary files. The statistical records identify the paired scenes so that aggregate count metrics and paired comparisons can be checked. The complete RGB/optical-flow arrays and trained checkpoints were retained locally for this study and are not deposited in a public repository in this version. Upstream model sources are cited separately; their meshes are not redistributed with the paper. No public release of assets without established redistribution permissions is implied. The availability scope limits independent regeneration of the full image-to-prediction pipeline from the paper package alone.

## AI assistance

AI assistance was used for implementation support, analysis-script development, literature organization, and English drafting. The numerical tables were generated from saved per-clip evaluation records and checked against frozen experiment artifacts. The schematic is an original vector drawing of the implemented information flow.

## References

[1] Joaquín Santoyo, Willy Azarcoya, Manuel Valencia, Alfonso Torres, and Joaquín Salas. Frequency analysis of a bumblebee (Bombus impatiens) wingbeat. Pattern Analysis and Applications, 19(2):487–493, 2016. URL: https://link.springer.com/article/10.1007/s10044-015-0501-3, doi:10.1007/s10044-015-0501-3.

[2] Ofir Levy and Lior Wolf. Live repetition counting. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 3020–3028, 2015. URL: https://openaccess.thecvf.com/content\_iccv\_2015/html/Levy\_Live\_Repet ition\_Counting\_ICCV\_2015\_paper.html, doi:10.1109/ICCV.2015.346.

[3] Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. Counting out time: Class agnostic video repetition counting in the wild. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10384–10393, 2020. URL: https://arxiv.org/abs/2006.15418, doi:10.1109/CVPR42600.2020.01040.

[4] Huazhang Hu, Sixun Dong, Yiqun Zhao, Dongze Lian, Zhengxin Li, and Shenghua Gao. TransRAC: Encoding multi-scale temporal correlation with transformers for repetitive action counting. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18991–19000, 2022. URL: https://arxiv.org/abs/2204 .01018, doi:10.1109/CVPR52688.2022.01843.

[5] Xinjie Li and Huijuan Xu. Repetitive action counting with motion feature learning. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 6485–6494, 2024. URL: https://openaccess.thecvf. com/content/WACV2024/html/Li\_Repetitive\_Action\_Counting\_With\_Motion\_Feature\_Learning\_WACV\_20 24\_paper.html, doi:10.1109/WACV57701.2024.00637.

[6] Zhang Nengbo, Hann Woei Ho, and Ye Zhou. MoCom: Motion-based inter-MAV visual communication using event vision and spiking neural networks, 2025. arXiv preprint. URL: https://arxiv.org/abs/2510.14770, arXiv:2510.14770, doi:10.48550/arXiv.2510.14770.

[7] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling, 2018. arXiv preprint. URL: https://arxiv.org/abs/1803.01271, arXiv: 1803.01271, doi:10.48550/arXiv.1803.01271.

[8] Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 36(6):51–63, 2019. URL: https://arxiv.org/abs/1901.09948, doi:10.1109/MSP.2019.2931595.

[9] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. URL: https://papers.nips.cc/paper\_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[10] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022. URL: https: //openreview.net/forum?id=R8sQPpGCv0.

[11] Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pages 5026–5033, 2012. doi: 10.1109/IROS.2012.6386109.

[12] Kevin Zakka, Yuval Tassa, and MuJoCo Menagerie Contributors. MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo. GitHub repository, 2022. Crazyflie 2 model from commit 8161bba264; accessed 15 September 2026. URL: https://github.com/google-deepmind/mujoco\_menagerie.

[13] Mostafa12d. RLFlapping: Course project for CS5180. GitHub repository, 2025. Version at commit 3577d0c41d, dated 18 November 2025; accessed 15 September 2026. URL: https://github.com/Mostafa12d/RLFlapping/tree/35 77d0c41d3d40e1f82ecdcc55e490e33a943c4f.

[14] Mintae Kim, Jiaze Cai, and Jack Leckert. flappy\_v2: PPO-based flapping wing MAV controller. GitHub repository, 2024. Version at commit d050e8b5c4, dated 15 March 2024; accessed 15 September 2026. URL: https://github.com/JiazeCai/flappy\_v2/tree/d050e8b5c4f46beec7088189efd668485e498ccc.

[15] Milkomedia. BIRD\_SIM: Bird model simulation in MuJoCo using MST aerodynamics. GitHub repository, 2026. Version at commit c5d5c3c123, dated 14 September 2026; accessed 15 September 2026. URL: https: //github.com/Milkomedia/BIRD\_SIM/tree/c5d5c3c123c0d648bed07205a5985725f8e07217.

[16] Gunnar Farnebäck. Two-frame motion estimation based on polynomial expansion. In Image Analysis, volume 2749 of Lecture Notes in Computer Science, pages 363–370. Springer, 2003. URL: https://link.springer.com/chap ter/10.1007/3-540-45103-X\_50, doi:10.1007/3-540-45103-X\_50.