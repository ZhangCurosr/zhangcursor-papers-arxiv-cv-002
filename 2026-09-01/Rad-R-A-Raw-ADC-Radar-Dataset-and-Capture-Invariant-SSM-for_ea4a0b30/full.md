# Rad-R: A Raw-ADC Radar Dataset and Capture-Invariant SSM for Hardware-Fault Diagnosis

Mainak Mallick Junghwan Yim Seung-Kyum Choi

Georgia Institute of Technology

mmallick7@gatech.edu jyim67@gatech.edu schoi@me.gatech.edu

## Abstract

Automotive mmWave radar can develop vibration, antenna misalignment, radome blockage, and receive-channel degradation that corrupt the signal before perception begins. Data for these faults are scarce because each condition must be induced and measured on physical hardware. We introduce Rad-R, a raw-ADC dataset captured with a 4-chip 77 GHz TI MMWCAS-RF-EVM cascade (192 virtual channels). Unlike existing raw-radar datasets, Rad-R pairs each recording with a controlled hardware fault at a calibrated severity, an independent physical severity measurement, andframe-synchronised IMU, temperature, GPS, and camera streams. Rad-R is a single-session dataset, so our generalisation claims are confined to a controlled cross-severity protocol in which train and test use physically distinct captures. A reproducible benchmark evaluates seven representative vision backbones and the proposed raw-IQ Mamba SSM (RadrNet) under within-clip, chirp-wise anytime, few-shot cross-capture, and controlled cross-severity protocols. Within-clip performance is nearsaturated (> 0.98 macro-F1), whereas cross-severity generalisation remains difficult: the absolute-phase RadrNet-DS falls to 0.49 macro-F1. RadrNet-DS-CI replaces absolute phase with per-frame-standardised magnitude and relative chirp-to-chirp phase and ranks first on the controlled benchmark (0.663 vs. 0.628 for the strongest RD-CNN; three seeds); the RadrNet family also leads on the anytime and few-shot budgets. A descriptive cross-modal analysis further finds that radar micro-Doppler covaries with independently measured IMU vibration energy (pooled Spearman ρ = 0.41 across conditions). The complete dataset and code will be released publicly under permissive licences.

## 1. Introduction

Millimetre-wave (mmWave) frequency-modulated continuous-wave (FMCW) radar is central to automotive perception [1, 2, 8] but degrades in deployment: vibration shifts alignment, partial radome blockage attenuates returns, individual Rx channels drift, and interference corrupts measurements. Existing public datasets either release raw analogue-to-digital-converter (ADC) samples under nominal operating conditions or provide processed tensors across varied scenes and weather [1–3, 5, 7]; none combines raw ADC with controlled hardware-fault annotations and a sensor-state evaluation suite. Rad-R fills that gap with (i) raw ADC from a 4-chip TI MMWCAS-RF-EVM cascade radar (Fig. 1; 12 Tx × 16 Rx = 192 virtual channels, 77 GHz) under physically induced faults, (ii) per-radarframe synchronised companion sensors (IMU, temperature T, GPS, camera), and (iii) a single-command evaluation harness with a Croissant-validated dataset card [37, 38]. Because these faults act on the per-channel phase and gain that range–Doppler (RD) processing averages away, and because a diagnosis is needed early in each frame, we pair the dataset with RadrNet, a state-space model (SSM) that consumes the raw IQ tensor directly and can classify from a prefix of chirps. Hardware-fault data are also inherently scarce: every example must be induced on real hardware. This constraint mirrors the limited-label regimes studied in fault diagnosis [44, 45] and industrial anomaly detection [46], motivating Rad-R as a benchmark for learning faults from limited labels.

## Contributions.

1. Rad-R dataset. Nine captures (∼27 k frames; 324 GB raw ADC) span healthy operation and four physically induced faults at two sensor-gated severities, with synchronised companion sensors, CC BY 4.0 licensing, and Croissant metadata.

2. RadrNet. A hierarchical 3-stage Mamba [31] processes raw complex in-phase/quadrature (IQ) samples with complex-aware encoding and multi-scale fusion; RadrNet-DS adds an RD-map CNN fused by an inputdependent gate.

3. Capture-invariant RadrNet. A capture-invariant encoding (per-frame-standardised magnitude plus relative chirp-to-chirp phase, which cancels constant global and per-channel phase offsets by construction) raises singlestream cross-severity F1 from 0.545 to 0.612; an ablation attributes most of this gain to magnitude standardisation (appendix). RadrNet-DS-CI exceeds the strongest RD-CNN by +0.035 mean macro-F1 and wins at each of three seeds.

4. Controlled benchmark. We compare RadrNet with seven vision backbones on two budget axes and a capture-disjoint cross-severity split. RadrNet-DS leads ResNet-18 at all six chirp budgets (k = 4–64) and leads DeiT-III from five labelled target frames per class.

## 2. Related Work

Radar datasets. RADIal [1] releases raw ADC with object boxes; K-Radar [2] provides 4D tensors across weather; CRUW [3], RadarScenes [4], RADIATE [9], and Dual Radar [6] provide processed RA or point-cloud data; CAR-RADA [7] provides RAD tensors, and RODNet [20] targets radar object detection. To our knowledge, no prior public radar dataset pairs raw ADC with hardware-fault labels (Table 1).

Radar fault, interference, and raw-ADC learning. Degraded sensing is safety-relevant in deployed ADAS [16], yet no public benchmark exposes how controlled radar hardware faults propagate into learned perception. This gap is analogous to the distribution shifts studied by WILDS [48], but at the sensor-hardware level. RIMformer [10] and RIME-Net [11] target interference mitigation on RD maps; FMCW degradation studies [12– 14] catalogue faults without a public raw-ADC benchmark; bearing-style diagnosis [15] does not model radar’s complex multi-channel structure. ADCNet [17] distils RD→ADC; Mamba-RODNet [18] applies Mamba to RD; and T-FFTRadNet [19] uses transformers on radar tensors. To our knowledge, RadrNet is the first hierarchical Mamba evaluated on complex raw IQ for both chirp-wise anytime and few-shot label-budget inference.

Streaming and anytime radar inference. SSMRad-Net [21] processes raw ADC samples with sample-wise and chirp-wise SSMs [33], while RadMamba [22] applies a Mamba SSM to radar micro-Doppler activity recognition. RECORD [23] uses a recurrent CNN for causal online detection, and RAVEN [24] formalises chirp-wise early exit with per-budget calibration. These works motivate our anytime / chirp-wise evaluation (Section 5.2). We adopt the paired-bootstrap and McNemar protocol of [25] for rigorous multi-seed comparisons.

## 3. The Rad-R Dataset

## 3.1. Hardware and Capture Setup

Rad-R uses a TI MMWCAS-RF-EVM [39] (Fig. 1; an exploded view of the rig is in Appendix B): a 4-chip

![](images/b459d7685a4f7e4e46e5dd7ed93c1dfa6eef19fe297cdc774ab3f4765ce0cd19.jpg)  
Figure 1. Rad-R in deployment. Top: the rig mounted on the test vehicle (front and side), behind the front grille. Bottom: the assembled sensor plate (front and back) showing the AWR1243 cascade radar, RealSense camera, BNO055 IMU, GPS, eccentricmass vibration motor, temperature probe, Raspberry Pi and 1 TB SSD. Captures are collected on private lots and permitted campus roads at low speed.

AWR1243 cascade at 77 GHz with 12 Tx and 16 Rx, yielding 192 virtual channels. Four DCA1000EVMs record the radar stream [40]. A 1/16-inch polypropylene radome is installed for all captures. A BNO055 IMU (∼33 Hz), two DHT22 temperature probes, u-blox NEO-M9N GPS, and two RealSense D435 camera streams (640×480 at ∼30 fps) are aligned to the radar frame index in synchronised HDF5 files. Figure 2 shows one radar frame of a misalignment capture together with every companion stream at that instant.

Chirp configuration and virtual-array geometry. Every Rad-R capture uses TI’s reference cascade-imaging chirp parameters (full table in the appendix): 256 ADC samples per chirp at an 8 Msps (mega-samples per second) sample rate, a 78.986 MHz/µs ramp giving 2.49 GHz effective bandwidth (0.060 m range resolution, 15.2 m maximum range), 64 loops per frame and a 100 ms frame period (∼10 Hz). The cascade runs in TDM-MIMO mode: within each loop the 12 transmit antennas are time-multiplexed across the 16 physical receive channels, so MIMO unmixing reconstructs a 192-element virtual array (16×12) whose angular resolution far exceeds that of any single chip. This dense virtual array carries independent per-receiver phase and amplitude structure in raw IQ that incoherent range– Doppler averaging discards, which is what makes the raw-IQ representation richer than the averaged RD map for fault classification. The four DCA1000EVMs stream raw int16 ADC samples (real/imag interleaved) to disk in TI’s standard <sub>\*</sub> data.bin/<sub>\*</sub> idx.bin format (one master, three slaves per session).

Table 1. Comparison of public automotive / mmWave radar datasets: raw-ADC availability, MIMO virtual channels, data format, task, hardware-fault and severity annotations, and companion sensors.
<table><tr><td>Dataset</td><td>Year</td><td>Raw ADC</td><td>MIMO (virt.)</td><td>Format</td><td>Task</td><td>Fault annot.</td><td>Sev.</td><td>Comp. sens.</td></tr><tr><td>CARRADA [7]</td><td>2020</td><td></td><td>12</td><td>RAD tensor</td><td>object detect.</td><td></td><td></td><td>cam</td></tr><tr><td>CRUW [3]</td><td>2021</td><td></td><td>12</td><td>RA map</td><td>object track.</td><td></td><td></td><td>cam</td></tr><tr><td>RadarScenes [4]</td><td>2021</td><td></td><td>n/r</td><td>point cloud</td><td>semantic seg.</td><td></td><td></td><td>cam, GPS</td></tr><tr><td>RADIal [1]</td><td>2022</td><td>了</td><td>192</td><td>ADC + RD + RA</td><td>object detect.</td><td></td><td></td><td>cam, lidar, GPS</td></tr><tr><td>K-Radar [2]</td><td>2023</td><td></td><td>n/r</td><td>4D tensor</td><td>object detect.</td><td>_</td><td></td><td>cam, lidar, GPS</td></tr><tr><td>Dual Radar [6]</td><td>2024</td><td></td><td>2304</td><td>4D tensor</td><td>object detect.</td><td></td><td>一</td><td>cam, lidar</td></tr><tr><td>Rad-R (ours)</td><td>2026</td><td>√</td><td>192</td><td>ADC + RD + RA</td><td>fault classification</td><td>4 types</td><td>2</td><td>cam, IMU, GPS, T, RH</td></tr></table>

Virtual channels = nominal Tx×Rx of the sensor MIMO array (n/r: not an imaging-MIMO array, or not reported); only RADIal and Rad-R release the raw per-channel ADC needed to access it. RA / RD / RAD: range–azimuth / range–Doppler / range–azimuth–Doppler; cam: camera; T / RH: temperature / relative humidity. <sup>‡</sup> K-Radar varies environmental conditions (rain, snow, fog) but does not induce hardware faults at the sensor.  
Rad-R per-capture dashboard — yaw2(snapshot t=217.0s, radar frame 2122; all panels at this instant)

![](images/81eb920992cbd33fd5a1dfccb055d22e1015f9ae1e095958da1b24d0a359c6a4.jpg)

![](images/17892775b3e814d26bf3ad6fd6e10478505d977d273a08043208c2b99bfdafa6.jpg)

![](images/650f7a6a121ad238eda6742fad09bb144112f1833fe5635f97a4e613d3ce6729.jpg)

![](images/d3ea162cbfc7298d01136c1c432d780353701529c2b6e766389f996cfd11b183.jpg)

![](images/300c76b452c632315a7845d1e14e8db2d5e905b2bb7a6f34fea7022a3fd16f7c.jpg)

![](images/62d0bc7cbc87bf8f90307d19e56cf24254192e93ae933c8ec5d4f464538a2c31.jpg)

![](images/979f12cb1e6a38140b05a50dbecc5300ce85bfcc805da4b57b51d1fe4d32d19b.jpg)  
t (s)

![](images/220e66ec047c9617f52a5dcfd0300cbde6722e7c93e5ea3a56c1f9cf0058744b.jpg)

![](images/c4023edf50ad1ac4277c3f18a95cb1637de4d880ce8f1b76867206c024b1a161.jpg)  
t (s)

![](images/463e83f8c7ecbd55f12b3d74b4661d0e210e5edbd3d404e9867ffa495244980c.jpg)  
t (s)  
Figure 2. Per-capture dashboard for the severe-misalignment capture $( \mathrm { y a w } 2 , 1 0 ^ { \circ }$ yaw) at radar frame 2122 (t = 217 s). Top left: chirpmean range profile, range–Doppler map (static-clutter cancellation for display only), micro-Doppler spectrum, and power across the 192 virtual channels at this frame. Top right: the two synchronised camera streams at the same timestamp (faces and licence plates blurred). Bottom: IMU acceleration magnitude, IMU temperature, DHT22 temperature/humidity, and GPS speed over the 5-minute capture; the dashed line marks this frame. Dashboards for the other eight captures are in Appendix C.1.

## 3.2. Fault Taxonomy and Induction Procedures

Four physically induced fault types are each captured at two severity levels (S1 mild, S2 severe), plus a healthy baseline (S0); fault induction is hardware-physical, not synthetic injection (Table 2). Each fault’s severity is gated by an independent physical sensor, not by the operator’s nominal setting: an eccentric-mass DC motor bolted to the radar bracket drives vibration at 20/40 Hz with the BNO055 IMU recording ground-truth tri-axial acceleration; a precision yaw jig with 1<sup>◦</sup> graduations sets misalignment to $5 ^ { \circ } / 1 0 ^ { \circ }$ , verified with $\mathrm { ~ a ~ } 0 . 1 ^ { \circ }$ digital inclinometer; a laser-cut clear polycarbonate sheet over the always-installed polypropylene radome produces 30%/60% blockage, calibrated against a corner-reflector dB drop; and 3M 1181 copper foil tape covering 6/16 or 10/16 Rx antenna patches induces Rx-channel degradation, with the per-channel dB drop confirmed by a single-tone calibration sweep before each capture. Because the induction is physical and the gate independent, the severity labels are objective rather than nominal. Thermal stress and RF interference are part of the dataset specification but deferred to future releases.

Range Doppler maps for each captured fault class (static-clutter cancellation applied for display; released RD is the raw map)  
![](images/6646787f3be319060e51cea9174a1b364614fb50ee4c219318ef8edf1f360bda.jpg)  
Figure 3. Range–Doppler maps for each fault × severity cell (columns: fault type; rows: severity). For display we apply static-clutter cancellation (slow-time mean subtraction) so scene structure is visible rather than the zero-Doppler leakage line; the released RD produc is the raw map. The per-fault signatures are physically interpretable but subtle (Sec. 3.3).

Table 2. Fault taxonomy and induction parameters captured in Rad-R.
<table><tr><td>Fault</td><td>Induction</td><td>S1</td><td>S2</td><td>Severity gate</td></tr><tr><td>Healthy</td><td>n/a</td><td>n/a</td><td>n/a</td><td>n/a</td></tr><tr><td>Vibration</td><td>Ecc. DC motor</td><td>20 Hz</td><td>40 Hz</td><td>IMU RMS</td></tr><tr><td>Misalign.</td><td>Yaw jig</td><td>5°</td><td>10°</td><td>Inclinometer</td></tr><tr><td>Blockage</td><td>PC on radome</td><td>30%</td><td>60%</td><td>Reflector dB</td></tr><tr><td>Rx degrad.</td><td>Cu foil tape</td><td>6/16 Rx</td><td>10/16 Rx</td><td>Cal sweep</td></tr></table>

## 3.3. Capture Protocol and Dataset Statistics

Each capture is a continuous 5-minute recording at ∼10 fps (∼3 k frames); the fault state is held throughout. Rad-R contains 9 capture runs (∼27 k frames, ∼324 GB raw ADC). For ML training we release a pre-processed cache of 1 800 frames (200 evenly sampled per capture, classbalanced 200 healthy / 400 per fault) with 224×224 RD maps (dB-scaled, per-frame normalised) and raw IQ at the canonical (loops, samples, virtual channels) layout. Companion sensors carry physically meaningful per-fault signatures (e.g. vibration 1.5× higher tri-axial gyro RMS; percapture traces in Appendix C), enabling severity regression and sensor-fusion research.

Signal-processing pipeline. The released RD maps are produced from raw ADC by a standard pipeline: windowed range/Doppler FFTs, non-coherent integration across the 192 virtual channels, dB conversion, per-frame normalisation, and a bilinear resize to 224 × 224 [41, 42] (recipe in the released code and dataset card). Range–azimuth and micro-Doppler maps are derived analogously and shipped for completeness, although the benchmark uses only RD.

Fault signatures in the released representations. Figure 3 shows representative range–Doppler maps for every fault × severity cell. The visual signatures are physically interpretable but subtle: vibration produces broadband Doppler spreading; blockage attenuates returns and elevates the noise floor; Rx-channel degradation perturbs the sidelobe structure while leaving a recognisable peak. Misalignment is near-invisible in RD: a few-degree yaw leaves range and Doppler essentially unchanged, acting only on the azimuth and per-channel phase that the channel average discards. This subtlety is precisely what motivates learned classifiers over hand-crafted features, and what makes the raw-IQ representation (which retains the per-channel structure averaged out of the RD map) valuable.

## 3.4. Evaluation Splits

Rad-R supports four protocols:

(1) Within-clip 80/20 temporal block. The first 80% of each capture trains the model and the last 20% tests it. The split contains no shared radar frames, but it does share scene, sensor, and configuration. We therefore treat it as a confounded upper bound rather than an independent generalisation test [1, 5].

(2) Anytime chirp budget. Models train on full 64-chirp frames from protocol (1) and are evaluated using only the first $k \in \{ 4 , 8 , 1 6 , 3 2 , 4 8 , 6 4 \}$ chirps (Section 5.2). This tests architectural behavior under truncated evidence, but inherits the within-capture confound.

(3) Few-shot cross-capture adaptation. A source-trained model receives $k \geq 1$ labelled frames per fault class from a held-out target capture and is evaluated on the remaining target frames (Section 5.3).

(4) Controlled cross-severity 2-fold. Training uses one severity of each fault and testing uses the other (Section 5.4); train and test therefore use physically distinct captures for every fault.

The single-day, single-device scope does not support rigorous held-out day, device, or scene evaluation; these axes are reserved for future releases.

## 4. Models and RadrNet

We benchmark models spanning the representation axis (RD map vs. raw complex IQ) and the inductive-bias axis (CNN, transformer [36], SSM): seven published RDmap backbones (ResNet-18, 11.2 M [34]; ViT-Small, 21.5 M [35]; ConvNeXt-V2 [26]; DeiT-III [27]; Swin-T [28]; PVTv2 [29]; EfficientNetV2 [30]), and the proposed RadrNet family. Every RD backbone uses a 1- channel stem and dual fault/severity heads. Severity is a gate level (S0/S1/S2) common to all faults, so a single shared severity head is used; it is trained as an auxiliary loss (weight 0.5; appendix) and does not enter the fault metrics. RadrNet is a hierarchical 3-stage Mamba [31, 32] on raw complex IQ tensors of shape (B, 2, C, S, R): batch, real/imaginary part, chirps C, ADC samples per chirp $S ,$ and virtual channels R. It patch-embeds a 4R-channel encoding (real, imaginary, magnitude, absolute phase) along C·S, then scans joint range/slow-time, Doppler, and crosschannel virtual-array structure. Multi-scale stage features feed dual fault and severity heads (Fig. 4). Four train-only augmentations (global phase rotation, per-Rx amplitude jit ter, Rx dropout, and per-frame standardisation) randomise capture-specific phase and gain. RadrNet-DS (∼5.1 M) adds a compact RD-map CNN and fuses the IQ and RD representations with a 2-way input-dependent gate.

Capture-invariant SSM. On the controlled cross-severity split (Section 5.4) the raw-IQ models above are the weakest cross-severity generalisers, despite leading within-clip. Absolute phase exposes global and per-channel calibration offsets that remain nearly constant within a capture; perframe magnitude scale can provide an additional capture cue. RadrNet-CI (∼4.1 M) keeps the 3-stage Mamba backbone but replaces real/imag+absolute-phase encoding with (i) per-frame-standardised magnitude and (ii) relative chirpto-chirp phase $( \Delta \phi )$ , encoded as (sin $\Delta \phi ,$ cos $\Delta \phi )$ (3R input channels). Differencing cancels constant global and per-channel phase offsets, while standardisation removes global magnitude scale; Section 5.4 and the appendix isolate the two components. RadrNet-DS-CI (∼4.7 M) fuses this invariant-IQ branch with a compact RD-CNN through the same 2-way input-dependent gate.

Formulation. Let $\boldsymbol { x } ~ \in ~ \mathbb { C } ^ { C \times S \times R }$ be one frame $( C { = } 6 4$ chirps, S=256 ADC samples, $R { = } 1 9 2$ virtual channels). Encoding. The capture-invariant encoding is the parameterfree map

$$
\begin{array} { r } { E ( x ) = \left[ \tilde { m } , \sin \Delta \phi , \cos \Delta \phi \right] \in \mathbb { R } ^ { 3 \times C \times S \times R } , } \end{array}\tag{1}
$$

$$
\tilde { m } = \frac { | x | - \mu _ { x } } { \sigma _ { x } } , \quad \Delta \phi _ { c , s , r } = \mathrm { w r a p } \big ( \angle x _ { c + 1 , s , r } - \angle x _ { c , s , r } \big ) ,
$$

with $\mu _ { x } , \sigma _ { x }$ taken over all $( c , s , r )$ , wrap(·) to $( - \pi , \pi ]$ and zero padding at $c { = } 0 .$ . For $x _ { c , s , r } ^ { \prime } = g e ^ { j \left( \phi _ { 0 } + \psi _ { r } \right) } x _ { c , s , r }$ (global gain $g > 0$ , global phase $\phi _ { 0 } .$ per-channel calibration phase $\psi _ { r } )$ $E ( x ^ { \prime } ) = E ( x )$ exactly; a per-channel gain $g _ { r }$ is not cancelled and is covered by the training-time Rx-amplitude jitter.

Tokens and SSM stages. A strided 1-D convolution over the flattened chirp×sample axis yields

$$
u _ { n } = W _ { e } \sec \bigl ( E ( x ) _ { [ : , 8 n : 8 n + 8 ] } \bigr ) + b _ { e } \in \mathbb { R } ^ { d } , \quad n = 1 , \ldots , N ,\tag{2}
$$

with $N = C S / 8 = 2 0 4 8$ and $d = 2 5 6 .$ , so all 3R encodings of eight neighbouring samples enter each token. Each stage stacks two pre-norm residual selective-SSM blocks [31],

$$
\begin{array} { r } { h \gets h + \mathrm { S S M } ( \mathrm { L N } ( h ) ) , \qquad s _ { t } = \bar { A } _ { t } s _ { t - 1 } + \bar { B } _ { t } h _ { t } , \ y _ { t } = C _ { t } s _ { t } , } \end{array}\tag{3}
$$

with input-dependent $\left( \Delta _ { t } , B _ { t } , C _ { t } \right)$ . Stage 1 scans u<sub>1:N</sub> (joint range×slow time); its output is reshaped to a $C \times$ $S / 8$ grid and stage 2 scans along c for each range patch (Doppler); stage 3 scans the $W _ { p }$ -projected per-patch means across range patches (cross-range, virtual-array context). With $f _ { k }$ the token mean after stage $k ,$

$$
\begin{array} { r l } & { \beta = \mathrm { s o f t m a x } \big ( \mathrm { M L P } ( [ \mathrm { L N } f _ { 1 } ; \mathrm { L N } f _ { 2 } ; \mathrm { L N } f _ { 3 } ] ) \big ) \in \mathbb { R } ^ { 3 } , } \\ & { z _ { \mathrm { I Q } } = \mathrm { M L P } \Big ( \sum _ { k } \beta _ { k } \mathrm { L N } f _ { k } \Big ) . } \end{array}\tag{4}
$$

RD stream and gate. The RD stream uses the released map

$$
\begin{array} { r } { D ( x ) = \mathrm { r e s i z e } _ { 2 2 4 } \Big ( 1 0 \log _ { 1 0 } \frac { 1 } { R } \sum _ { r } \big | \mathcal { F } _ { c } \mathcal { F } _ { s } ( w \odot x ) \big | _ { r } ^ { 2 } \Big ) } \end{array}\tag{5}
$$

(Hann windows w, min–max normalised per frame) and $z _ { \mathrm { R D } } = \mathrm { C N N } ( D ( x ) ) \in \mathbb { R } ^ { d }$ (four strided convolutions, 32→ 64 → 128 → 256 channels, global average pool). An inputdependent gate $\alpha =$ softmax $\big ( \mathrm { M L P } ( [ \mathrm { L N } z _ { \mathrm { R D } } ; \mathrm { L N } z _ { \mathrm { I Q } } ] ) \big ) \ \in$ $\mathbb { R } ^ { 2 }$ fuses the two streams,

$$
z = \mathrm { L N } \big ( \mathrm { G E L U } ( W _ { o } ( \alpha _ { \mathrm { R D } } \mathrm { L N } z _ { \mathrm { R D } } + \alpha _ { \mathrm { I Q } } \mathrm { L N } z _ { \mathrm { I Q } } ) ) \big ) ,\tag{6}
$$

and two linear heads on z give the 5-way fault and 3-way severity logits, trained with $\mathcal { L } ~ = ~ \mathrm { C E } _ { \mathrm { f a u l t } } + 0 . 5 \mathrm { C E } _ { \mathrm { s e v } } ,$ RadrNet-CI feeds $z _ { \mathrm { I Q } }$ to the heads directly.

![](images/5a6bfafbdd96fa227406a487c4a2c071c60d465035ea7e76046fce99457ab077.jpg)  
Figure 4. RadrNet-DS-CI architecture. The capture-invariant IQ stream encodes per-frame-standardised magnitude m˜ and relative chirp to-chirp phase (sin ∆ϕ, cos ∆ϕ), then processes range, Doppler, and virtual-array structure with a 3-stage hierarchical Mamba [31] and multi-scale fusion. An input-dependent 2-way gate combines the IQ representation with a compact RD-CNN representation before the 5-way fault and 3-way severity heads. RadrNet-CI omits the RD stream.

## 5. Experiments and Results

Setup. All models train on the same 1,800-frame cache with AdamW (lr 10<sup>−4</sup>, weight decay 10<sup>−4</sup>), cosine schedule, gradient-norm clip 1.0, batch size 16, and 20 epochs (a subset of the cross-severity runs use 25; see appendix). The saved runs select the epoch with maximum macro-F1 on the evaluation split; because no separate validation partition was used, the within-clip and cross-severity scores are oracle-selected upper bounds. Few-shot adaptation instead uses a fixed 30-step schedule. Fault macro-F1 is the primary metric; fault/severity accuracy and 15-bin ECE are retained in the released result records. Full hyperparameters, metrics, and compute are in the appendix.

## 5.1. Within-Clip Split as Context

Within-clip split is near-saturated. The within-clip 80/20 split is the regime on which the anytime models are trained (Section 3.4), but it does not discriminate architectures: under the identical recipe of Section 4, the strongest published backbones reach 0.974 macro-F1 (DeiT-III, Swin-T) and minimal-inductive-bias raw-IQ baselines (a 0.7 M 1D-CNN, a 6.8 M IQ-transformer) reach 0.989–0.994. Because this split shares scene and sensor, we report it only as a contextual upper bound. Generalisation claims rely on the cross-capture and controlled cross-severity protocols; the anytime experiment tests behavior under a chirp budget within the same confounded split. The released records additionally include 15-bin ECE [43], which matters when confidence influences a safety decision.

## 5.2. Anytime / Chirp-Wise Inference

Protocol. The chirp-budget axis compares RadrNet-DS with a ResNet-18 RD-map baseline. Both train on full 64- chirp frames from the within-clip 80/20 split (Section 3.4) for 20 epochs at each of five seeds. At evaluation, each model receives only the first $k ~ \in ~ \{ 4 , 8 , 1 6 , 3 2 , 4 8 , 6 4 \}$ chirps: RadrNet-DS consumes the truncated raw IQ directly, whereas ResNet-18 receives a fixed-size RD input recomputed after chirps $k , \ldots , 6 3$ are set to zero. Significance follows [25]: a one-sided paired bootstrap (n=2000, H<sub>0</sub>: $\mathrm { F 1 } _ { \mathrm { R a d r N e t - D S } } \leq \mathrm { F 1 } _ { \mathrm { R e s N e t } } )$ and a two-sided McNemar exact test on 1 800 pooled predictions (five seeds × 360 frames) at each k. Figure 5 (left) plots the curves; the full table is in the appendix.

Motivation. The protocol builds on recent efficient radar sequence models: sample-wise and chirp-wise SSM processing [21], micro-Doppler-oriented Mamba modeling [22], causal online recurrent detection [23], and calibrated chirp-wise early exit [24].

Findings. RadrNet-DS attains higher F1 than ResNet-18 at every k (paired-bootstrap $p \ < \ 0 . 0 5$ throughout). The gap is largest at small k (∆F1 stays in [+0.594, +0.683] for $k \leq 3 2$ , where ResNet-18 sits near the majority-class macro-F1) and narrows to +0.035 at the full k=64 frame, still significant (McNemar $p = 5 . 9 { \times } 1 0 ^ { - 5 } )$ . This pattern is consistent with the selective scan integrating truncated slow-time evidence more effectively than an RD-CNN operating on a poorly resolved, zero-padded Doppler spectrum; it does not by itself isolate the causal contribution of any

![](images/222e3ea54e7a03b17e41a3015f01ac37e307e5038ea86cdf7c9ca4376c84f672.jpg)

![](images/6713c7625bd7b22fda9c9b21e045863522c80fdb70cacc11261374d44a684860.jpg)  
\* p < 0.05 for RadrNet-DS vs. ResNet-18 (tests specified in text)  
Figure 5. Two budget axes (mean ± std bands). (Left) Chirp budget (anytime). Within-clip 80/20 MIMO split, five seeds; stars mark paired-bootstrap $p < 0 . 0 5$ at each k (test defined in the text). (Right) Label budget (few-shot). Cross-capture adaptation; stars mark unpaired-bootstrap $p < 0 . 0 5$ for RadrNet-DS over the ResNet-18 RD-CNN.

single component [21–24].

Latency. On an RTX A4000 (fp32), RadrNet-DS takes 5.5– 6.2 ms/frame and ResNet-18 takes 2.1–4.6 ms/frame. The roughly 2× wall-time buys +0.5–0.7 macro-F1 at small k while remaining below the 100 ms frame deadline.

## 5.3. Few-Shot Cross-Capture Adaptation

Protocol. A complementary label-budget axis to the chirp budget of Section 5.2. Each base model is trained on the source (one-severity) captures for 15 epochs; at test we draw k labelled frames per fault class from the held-out target capture, fine-tune for 30 steps at $\mathrm { l r } { = } 1 0 ^ { - 4 }$ (identical recipe across models), and evaluate on the remaining target frames. We sweep $k \in \{ 1 , 5 , 1 0 , 2 0 \}$ over 2 folds × 3 support draws per seed, comparing RadrNet-DS against ResNet-18 (the canonical RD-CNN) and DeiT-III (the strongest published backbone). As models draw independent support sets, significance uses a one-sided unpaired bootstrap (n=2000). RadrNet-DS and ResNet-18 use three seeds (18 fold×draw observations per k); DeiT-III and the capture-invariant variants are single-seed. The full per-budget table is in the appendix.

Findings. At k=1 RadrNet-DS is statistically tied with the ResNet-18 RD-CNN (0.584 vs. 0.574, p=0.335). From k=5 it leads both competitors (0.755 vs. ResNet’s 0.562, $p \ < \ 1 0 ^ { - 3 }$ , and DeiT-III’s single-seed 0.711). Its margin over DeiT-III is modest $( + 0 . 0 4 \mathrm { { - } 0 . 0 5 \mathrm { { F 1 } ) } }$ . From k=5 to 20, RadrNet-DS-CI rises from 0.732 to 0.761, while three-seed RadrNet-DS rises from 0.755 to 0.785; the invariant variant trades some few-shot adaptability for cross-severity gains.

## 5.4. Controlled Cross-Severity Generalisation

Protocol. The within-clip and anytime protocols share scene and sensor between train and test. To control this shortcut [47], we train on one severity of each fault and test on the other (2 folds), so every fault uses physically distinct train and test captures [48]. Result. Figure 6 reports fault macro-F1: off-the-shelf baselines fall from 0.71–0.97 within-clip $\mathrm { { t o } \leq 0 . 6 3 }$ , and the ranking inverts. RD-map CNNs lead the absolute-phase raw-IQ models (RadrNet 0.545, RadrNet-DS 0.485). The capture-invariant encoding raises the single-stream RadrNet-CI to 0.612±0.015, while RadrNet-DS-CI ranks first at $0 . 6 6 3 \pm 0 . 0 0 9 , + 0 . 0 3 5$ above ResNet-18 (0.628±0.015) and higher at each of three seeds. The largest gain occurs in the severe→mild fold (appendix). An encoding ablation (Appendix H.4) shows that magnitude standardisation alone reaches $0 . 6 3 1 \pm 0 . 0 2 8$ and relative phase alone $0 . 3 9 9 { \pm } 0 . 0 2 5$ : relative phase does not raise the mean but lifts the harder severe→mild fold $\left( 0 . 5 3 8 \right.$ 0.578) through blockage and Rx-degradation recall, and the per-fault breakdown there also explains why the RD branch hurts the absolute-phase model $( 0 . 5 4 5  0 . 4 8 5 )$ yet helps the invariant one $( 0 . 6 1 2  0 . 6 6 3 )$ . Substantial headroom remains: more than 0.3 F1 is lost relative to within-clip performance. A descriptive cross-modal analysis in the appendix shows that radar micro-Doppler covaries with independently measured IMU vibration energy across the three captured conditions (pooled Spearman $\rho = 0 . 4 1 )$ ).

Anytime under cross-severity. Under partial-chirp budgets (full values in the appendix), the two invariant models trade off: RadrNet-DS-CI is substantially weaker at small budgets (0.30 at k=4, 0.601 at k=64), whereas RadrNet-CI reaches 0.479 from only 4 of 64 chirps. The 0.601 endpoint comes from the separate prefix-evaluation harness with flat seed×fold averaging and final-epoch weights, and is therefore not interchangeable with the 0.663 full-frame aggregate above; it is, however, a selection-free estimate, and at k=64 the ordering is unchanged: RadrNet-DS-CI 0.601, ResNet-18 0.591, RadrNet-CI 0.580, RadrNet-DS 0.441 (three seeds for the CI models, one for the other two; Table S6). Neither invariant model dominates across budgets; the absolute-phase RadrNet-DS loses its within-clip anytime advantage and trails ResNet-18 at k=64.

![](images/6e6d32712466239791de848a7bf415d6367ae4f217277e08c5adac39bf26f6f0.jpg)  
Figure 6. Controlled cross-severity 2-fold benchmark: fault macro-F1 per model (three-seed mean; thin caps are std over seeds). Colour distinguishes our RadrNet variants (captureinvariant; absolute-phase raw-IQ) from off-the-shelf backbones. Per-model values with input modality are tabulated in the appendix.

## 6. Conclusion

Rad-R is the first radar dataset to pair raw ADC with controlled, physically induced hardware faults at independently gated, calibrated severities and per-frame synchronised companion sensors. Its controlled cross-severity benchmark exposes a failure that within-clip evaluation hides: absolute-phase raw-IQ SSMs are vulnerable to capturespecific shortcuts and decline sharply across severities. The capture-invariant encoding removes constant phase offsets and global gain by construction—with magnitude standardisation, not relative phase, accounting for most of the gain— and RadrNet-DS-CI is the strongest model on the controlled challenge (0.663 vs. 0.628 for the best RD-CNN). Substantial headroom remains (> 0.3 F1 below the within-clip regime), and the capture-invariant models trade some calibration for this robustness (ECE ≈ 0.17–0.21 vs. ≈ 0.14 for ResNet-18), so they require recalibration before deployment. Rad-R is single-session, and the saved within-clip and cross-severity checkpoints are oracle-selected on their evaluation splits; both limitations make the current numbers upper bounds rather than deployment estimates. The complete dataset and code will be released publicly under permissive licences.

## Acknowledgements

This work was supported by the Korea Evaluation Institute of Industrial Technology (KEIT) grant funded by the Ministry of Trade, Industry & Energy (MOTIE) of the Republic of Korea (No. RS-2024-00448797).

## References

[1] Rebut, J., Ouaknine, A., Perez, P., et al. Raw high-definition´ radar for multi-task learning. CVPR, 2022.

[2] Paek, D.-H., Kong, S.-H., et al. K-Radar: 4D radar object detection for autonomous driving in various weather conditions. NeurIPS Datasets & Benchmarks, 2022.

[3] Wang, Y., Jiang, Z., et al. Rethinking of radar’s role: A camera-radar dataset and systematic annotator via coordinate alignment. CVPRW, 2021.

[4] Schumann, O., Hahn, M., Scheiner, N., et al. RadarScenes: A real-world radar point cloud dataset for automotive appli cations. Fusion, 2021.

[5] Kramer, A., Harlow, K., Williams, C., Heckman, C. ColoRadar: The direct 3D millimeter wave radar dataset. IJRR, 2021.

[6] Wei, J., et al. Dual Radar: A dataset built upon dual 4D radars for object detection. Scientific Data, 2024.

[7] Ouaknine, A., Newson, A., Rebut, J., et al. CARRADA dataset: Camera and automotive radar with range-angle-Doppler annotations. ICPR, 2020.

[8] Caesar, H., Bankiti, V., Lang, A., et al. nuScenes: A multi modal dataset for autonomous driving. CVPR, 2020.

[9] Sheeny, M., De Pellegrin, E., et al. RADIATE: A radar dataset for automotive perception in bad weather. ICRA, 2021.

[10] Chen, R., et al. RIMformer: An end-to-end transformer for radar interference mitigation. ICASSP, 2024.

[11] Wang, T., Sui, X., Lu, K. RIME-Net: An efficient deep en coder for radar interference mitigation. IEEE Sensors, 2023.

[12] Tinselboer, A., Vlek, P., et al. Survey of fault detection in FMCW automotive radar systems. IEEE T-IV, 2023.

[13] Linge, S. P. M., et al. Vibration-induced phase noise in MMIC FMCW radar transceivers. IEEE T-MTT, 2020.

[14] Roos, F., Bechter, J., et al. Misalignment monitoring for au tomotive radar from field data. IEEE T-IV, 2024.

[15] Smith, W. A., Randall, R. B. Rolling element bearing diagnostics using the Case Western Reserve University data: a benchmark study. Mech. Syst. Signal Process., 2015.

[16] National Highway Traffic Safety Administration. Initial standing general order on crash reporting: ADAS sensor in cident analysis. NHTSA report DOT-HS-813-415, 2023.

[17] Liu, Y., Cui, T., et al. ADCNet: Learning from raw automotive radar ADC data via cross-representation distillation. IEEE T-IV, 2023.

[18] Park, J., Kim, B.-H., et al. MambaRODNet: A selective state-space model for radar object detection. IEEE Sensors, 2024.

[19] Yang, T., et al. T-FFTRadNet: Transformer architectures on cascaded radar tensors. IEEE Access, 2024.

[20] Wang, Y., Jiang, Z., et al. RODNet: Radar object detection using cross-modal supervision. WACV, 2021.

[21] Sen, A., Mohammad, M.S., Mukhopadhyay, S. SSMRadNet: A sample-wise state-space framework for efficient and ultralight radar segmentation and object detection. WACV, 4365– 4374 (2026).

[22] Wu, Y., Fioranelli, F., Gao, C. RadMamba: Efficient human activity recognition through radar-based micro-Doppleroriented Mamba state-space model. IEEE Trans. Radar Syst. (2025). doi:10.1109/TRS.2025.3648848.

[23] Boulch, A., Cherenkova, K., et al. RECORD: A recurrent convolutional radar object detector for streaming inference. IEEE T-ITS, 2024. arXiv:2212.11172.

[24] Sen, A., Mohammad, M.S., Mukhopadhyay, S. RAVEN: Radar adaptive vision encoders for efficient chirp-wise object detection and segmentation. CVPR, 17938–17947 (2026).

[25] Du, W., et al. When +1% is not enough: a paired bootstrap protocol for evaluating small improvements. arXiv:2511.19794, 2025.

[26] Woo, S., Debnath, S., Hu, R., et al. ConvNeXt V2: Codesigning and scaling ConvNets with masked autoencoders. CVPR, 2023.

[27] Touvron, H., Cord, M., Jegou, H. DeiT III: Revenge of the´ ViT. ECCV, 2022.

[28] Liu, Z., Lin, Y., Cao, Y., et al. Swin Transformer: Hierarchical vision transformer using shifted windows. ICCV, 2021.

[29] Wang, W., Xie, E., Li, X., et al. PVT v2: Improved baselines with Pyramid Vision Transformer. Computational Visual Media, 2022.

[30] Tan, M., Le, Q. EfficientNetV2: Smaller models and faster training. ICML, 2021.

[31] Gu, A., Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. COLM, 2024 (arXiv 2312.00752, 2023).

[32] Dao, T., Gu, A. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. ICML, 2024.

[33] Gu, A., Goel, K., Re, C. Efficiently modeling long sequences´ with structured state spaces. ICLR, 2022.

[34] He, K., Zhang, X., Ren, S., Sun, J. Deep residual learning for image recognition. CVPR, 2016.

[35] Dosovitskiy, A., et al. An image is worth 16×16 words: Transformers for image recognition at scale. ICLR, 2021.

[36] Vaswani, A., et al. Attention is all you need. NeurIPS, 2017.

[37] Akhtar, M., Benjelloun, O., et al. Croissant: A metadata format for ML-ready datasets. ICLR Data-Centric AI Workshop, 2024.

[38] Pushkarna, M., Zaldivar, A., Kjartansson, O. Data cards: Purposeful and transparent dataset documentation. FAccT, 2022.

[39] Texas Instruments. MMWCAS-RF-EVM and MMWCAS-DSP-EVM: Cascade imaging radar EVM user’s guide. SWRA574, 2020.

[40] Texas Instruments. DCA1000EVM: Capture card for use with mmWave radar evaluation modules. SPRUIJ4A, 2019.

[41] Richards, M. A. Fundamentals of Radar Signal Processing, 2nd ed. McGraw-Hill, 2014.

[42] Harris, F. J. On the use of windows for harmonic analysis with the discrete Fourier transform. Proc. IEEE, 1978.

[43] Guo, C., Pleiss, G., Sun, Y., Weinberger, K. On calibration of modern neural networks. ICML, 2017.

[44] Zhang, A., Li, S., Cui, Y., Yang, W., Dong, R., Hu, J. Limited data rolling bearing fault diagnosis with few-shot learning. IEEE Access, 2019.

[45] Wang, Y., Yao, Q., Kwok, J. T., Ni, L. M. Generalizing from a few examples: A survey on few-shot learning. ACM Com puting Surveys, 2020.

[46] Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. MVTec AD: A comprehensive real-world dataset for unsupervised anomaly detection. CVPR, 2019.

[47] Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F. A. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2020.

[48] Koh, P. W., et al. WILDS: A benchmark of in-the-wild distribution shifts. ICML, 2021.

[49] Gebru, T., Morgenstern, J., et al. Datasheets for datasets. Comm. ACM, 2021.

# Supplementary Material

The following appendices contain the supplementary material referenced in the main text; figures and tables carry an “S” prefix, while references are shared with the main text.

Table S1. MMWCAS-RF-EVM chirp and frame configuration used for every Rad-R capture.
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Start freq.</td><td>77 GHz</td><td>Loops/frame (M)</td><td>64</td></tr><tr><td>Freq. slope</td><td>78.99 MHz/µs</td><td>Frame period</td><td>100 ms</td></tr><tr><td>ADC samples (N)</td><td>256</td><td>Cascade devices</td><td>4</td></tr><tr><td>ADC sample rate</td><td>8Msps</td><td>Total Rx</td><td>16</td></tr><tr><td>TDM-MIMO Tx</td><td>12</td><td>Virtual channels</td><td>192</td></tr><tr><td>Eff. bandwidth</td><td>2.49 GHz</td><td>Range res.</td><td>0.060 m</td></tr><tr><td>Max range</td><td>15.2m</td><td>Active Rx/device</td><td>4</td></tr></table>

## A. Detailed Radar Configuration

The exact chirp configuration (Table S1) follows Texas Instruments’ reference cascade-imaging configuration [39] and is stored verbatim in each capture’s <sub>\*</sub>.mmwave.json file. The DCA1000EVM cards stream interleaved complex int16 ADC samples to TI’s standard <sub>\*</sub>\_data.bin and <sub>\*</sub>\_idx.bin files, with one master and three slave devices per session. Released range–Doppler maps use standard FMCW processing [41] with Hann windows [42]; analogous pipelines produce the range–azimuth and micro-Doppler products. The benchmark uses only RD and raw IQ.

## B. Physical Fault Induction

Figure S1 shows the sensor rig in exploded view, including the misalignment, vibration and blockage fixtures. Each fault is produced by a physical fixture on the real radar, and its severity is set against an independent instrument rather than a nominal dial. Figure S2 shows the three fixtures whose effect is directly visible on the hardware, each at the healthy, mild (S1) and severe (S2) settings.

Vibration is the one fault that leaves no before-and-after photograph, because its signature is mechanical motion rather than a fixed obstruction. An eccentric-mass DC motor is bolted to the radar bracket and shakes the sensor while it streams. Severity is set by the drive frequency, 20 Hz for the mild setting (S1) and 40 Hz for the severe setting (S2), and is verified against the onboard BNO055 IMU rather than a dial: the IMU records ground-truth tri-axial acceleration, and a capture is accepted only when its RMS acceleration falls in the band recorded for the target setting. The disturbance therefore appears in the IMU and radar micro-Doppler traces (Appendix I quantifies this agreement) but not in a still image of the hardware, which is why it has no panel in Figure S2.

## C. Sample Radar Visualizations

The main paper presents the per-fault range–Doppler grid and one per-capture dashboard (yaw2). Here we add an expanded per-fault view (Fig. S3), the synchronised companion-sensor traces (Fig. S4), and per-capture dashboards for the remaining eight recordings (Figs. S5–S12). Each dashboard combines a range profile, range–Doppler map, micro-Doppler spectrum, per-channel power, two synchronised camera streams, and an IMU/temperature/GPS strip.

## C.1. Per-Capture Dashboards

For each capture (the yaw2 dashboard is in the main text), we include a multi-sensor dashboard generated from the raw ADC and ROS2 bag data. The upper-left 2×2 block contains the chirp-mean range profile, a display-normalised range–Doppler map, the Doppler power profile, and power across the 192 virtual channels. The upper-right block shows the monochrome and colour camera streams at the radar timestamp marked on the lower sensor strip; faces and licence plates are Gaussian-blurred. The lower strip contains IMU acceleration magnitude, IMU temperature, DHT22 temperature and humidity, and GPS speed over the radar capture window. Vibration elevates the IMU trace, blockage depresses the near-range return, and degraded receivers create dips in per-channel power. Static misalignment is less visible in these power views because its principal signature lies in azimuth and channel phase.

## D. Datasheet for Rad-R

Following Gebru et al. [49] and data-card guidance [38], we provide a datasheet for Rad-R. The released Croissant 1.0 metadata encodes the corresponding machine-readable fields (Section E).

## Motivation

For what purpose was the dataset created? Rad-R was created to study how automotive-radar hardware faults propagate through the signal-processing chain into learned models, and to support fault-aware radar perception. Existing datasets either provide raw ADC under nominal operation or processed representations across varied scenes and weather; none pairs raw ADC with controlled hardwarefault conditions and independently measured severities.

Who created the dataset and on behalf of which entity? The dataset was created by the authors of this paper (affili-

![](images/e5d69065eff1c28cbd4bc6a587ff3b067501d37e6438f1533fd5ecdc16159c70.jpg)  
Figure S1. The Rad-R sensor rig (exploded view). A 4-chip TI MMWCAS-RF-EVM cascade radar (12 Tx × 16 Rx, 77 GHz) sits behind an always-installed 1/16-inch polypropylene radome, rigidly co-mounted with a Bosch BNO055 IMU, two DHT22 temperature probes, a u-blox NEO-M9N GPS, and an Intel RealSense D435 camera; a Raspberry Pi and 1 TB SSD log every stream to synchronised HDF5. A yaw-adjustment ring, an eccentric-mass vibration motor, and a removable radome panel are the misalignment, vibration, and blockage fixtures, respectively. All companion sensors are time-aligned to the radar frame index.

ations on the title page).

Who funded the creation of the dataset? This work was supported by the Korea Evaluation Institute of Industrial Technology (KEIT) grant funded by the Ministry of Trade, Industry & Energy (MOTIE) of the Republic of Korea (No. RS-2024-00448797). The funder had no editorial influence over the dataset content or the reported conclusions.

## Composition

What do the instances that comprise the dataset represent? Each instance is one radar frame (∼100 ms wallclock) containing the raw complex IQ tensor from the 4- chip cascade radar, derived range–Doppler map, and timealigned readings from each companion sensor stream at the moment of that radar frame. Frames are grouped into 9 capture runs of 5 minutes each (∼3,000 frames per run).

How many instances are there in total? Rad-R contains 9 capture runs covering healthy + 4 fault types × 2 severity levels each. The training cache subsamples 200 frames per capture (1,800 frames total). The complete raw archive comprises ∼26,800 frames. A compact representative data sample accompanies the code release, and the complete raw archive will be released in a persistent public repository.

Does the dataset contain all possible instances or is it a sample? It is a sample: one representative capture per (fault, severity) cell, in a single outdoor / lab scene with one vehicle.

What data does each instance consist of? Each radar frame contains (1) raw complex IQ with M = 64 chirps, N = 256 ADC samples, R = 16 physical receivers, and C<sub>tx</sub> = 12 time-multiplexed transmitters in TI’s interleaved int16 layout; (2) a pre-computed 224×224, dB-scaled range–Doppler map; and (3) indices and timestamp offsets for the synchronised IMU, DHT22, GPS, and dual-camera streams.

![](images/2311f3d3109caf2286de0423e717585e37098726788783b8b1e9bb1b93996209.jpg)  
Figure S2. Physical fault induction. Top: misalignment, the radar enclosure is yawed on a precision jig by $5 ^ { \circ }$ then $1 0 ^ { \circ }$ (verified with a digital inclinometer). Middle: blockage, a clear polycarbonate sheet covers 30% then 60% of the radar aperture (calibrated against a corner-reflector dB drop). Bottom: receive-channel degradation, copper foil tape disables 6/16 then 10/16 of the Rx antennas (confirmed by a single-tone calibration sweep). The leftmost panel in each row is the healthy baseline. Vibration is omitted here because its effect is dynamic rather than static; it is described in the text below.

Per-fault visualizations: RD map (top, static-clutter removed) / zero-Doppler range profile (mid) / raw IQ slow-time envelope (bottom). Real captures, ego moving.

![](images/735ebac2779a8dd6837b90ec0e1506d0e01f03e20a833ed61eb14e78b61925d9.jpg)  
Figure S3. Expanded radar visualisation for each of the five captured classes (rows). From left: the range–Doppler map (static-clutter cancellation applied only for display), the zero-Doppler range profile, and the slow-time envelope of the raw-IQ tensor.

Is there a label or target associated with each instance? Yes: a fault class label (5-way: healthy, vibration, misalignment, blockage, Rx degradation) and a severity label (3-way: S0 healthy, S1 mild, S2 severe). Labels are determined by the physical fault induction protocol, not by post-hoc human annotation.

Is any information missing from individual instances? No required radar fields are intentionally omitted. After TDM-MIMO unmixing, the cache stores both a single-Tx 16-channel slab and the full 192-channel virtual-array slab so that experiments can use either representation. Companion-sensor availability and timestamp offsets are recorded per frame.

Does the dataset contain data that might be considered confidential or personally identifiable? The two co-recorded cameras (camera2 monochrome, camera4 BGR color) may incidentally capture pedestrians or license plates during outdoor captures. Every released camera frame (camera2 and camera4) has been processed through our automated PII-redaction pipeline that detects and Gaussian-blurs humanfaces (using a YOLO-basedface detector, σ=21px on the bounding box) and vehicle license plates (license-plate detector, same blur kernel) before upload. The redacted (<sub>\*</sub> image raw sensored) directories are the ones referenced from the synced HDF5; the unredacted originals never leave the authors’ workstation. We further manually spot-checked a uniform random sample of 5% of frames per capture to verify no detector misses slipped through, and re-blurred a small handful of edge cases (partial occlusions, oblique viewing angles) by hand.

## Collection Process

How was the data acquired? mmWave Studio captured raw <sub>\*</sub>\_data.bin files via the four DCA1000EVM cards. A Raspberry Pi 4 ran collect\_sensors.py to log the BNO055 IMU at ∼33 Hz, two DHT22 probes at ∼1 Hz, the NEO-M9N GPS, and two cameras (monochrome and BGR colour) at ∼30 fps each. Clocks were NTP-synchronised before each session and aligned at radar-frame level during post-processing.

What mechanisms or procedures were used to collect the data? Each session followed a fixed protocol: power-on and 10-minute thermal stabilization; healthy reference capture; controlled fault-induction sequence (one fault per run, severity held constant for the 5-minute capture); healthy reference recapture; per-clip metadata logging; per-file SHA-256 verification.

Who was involved in the data collection process? Two trained operators from the authors’ research group. The code release includes the fault-induction SOP (docs/

![](images/3ca73ae081728e097e53ed129c6598e2391a59cc3c2221e9a94a2caf3f2568dc.jpg)  
Figure S4. Synchronised companion-sensor traces, aligned to the radar frame index; each curve is one of the nine capture runs (thick: 9-frame rolling median; thin: raw). Panels: IMU AC acceleration, IMU gyro magnitude, board temperature, and GPS speed.

FAULT\_INDUCTION\_SOP.md) and printable session checklist (docs/FAULT\_INDUCTION\_CHECKLIST. pdf).

Over what timeframe was the data collected? Rad-R was collected on a single day in May 2026. Future releases will broaden the day/device/scene coverage.

Were any ethical review processes conducted? No human subjects research; no IRB review required. Lab safety review was conducted for the radar emission setup, the resistive heater (planned for future releases), and the secondary radar interferer (planned for future releases).

## Preprocessing, Cleaning, and Labeling

Was any preprocessing/cleaning/labeling of the data done? Raw ADC is stored without preprocessing. The released training cache contains derived range–Doppler maps and the raw IQ slab, both produced by scripts/build\_ training\_cache.py. Labels are physical-protocol labels, not annotation labels.

Was the “raw” data saved in addition to the preprocessed data? Yes. The raw mmWave Studio output is retained for every capture in TI’s exact <sub>\*</sub> data.bin format. Because a single raw capture is ∼37 GB, the code release includes compact representative raw-IQ excerpts together with the loader and preprocessing code; the complete raw archive will be released in a persistent public repository.

## Uses

What tasks has the dataset been used for? In this paper: within-clip fault classification, severity classification, chirp-wise anytime inference, few-shot cross-capture adaptation, and confusion-matrix analysis. Additional supported tasks include severity regression from the percapture sensor measurements, fault-aware sensor fusion, raw-IQ self-supervised pretraining, and signal-processingpipeline benchmarking.

Are there tasks for which the dataset should not be used? Rad-R should not be used to train safety-critical production radar-monitoring systems without extensive infield validation. The single-day single-device capture scope is too narrow for production deployment.

## Distribution and Maintenance

How is the dataset distributed? The code release contains training and evaluation code, result JSONs, Croissant metadata, and a compact PII-redacted data sample. The complete ∼324 GB raw archive will be released in a persistent public repository under CC BY 4.0. The code license is MIT, with the matlab to python/ subdirectory under BSD-3-Clause to preserve TI’s source license.

Will the dataset be updated? Yes. Future releases aim to broaden Rad-R along several axes (additional capture days, devices and scenes, further fault and interference modalities, and additional severity levels). Such updates will use tagged, persistent public releases with versioned identifiers.

## E. Croissant 1.0 Metadata and Responsible-AI Fields

The code release includes Croissant 1.0 metadata [37] in croissant.json. It covers file objects, file sets, record sets, and field types, plus the minimal Responsible-AI block [38]: collection and annotation protocols, release maintenance, personal information, limitations, biases, intended uses, social impact, data sources, provenance, and preprocessing. The file passes the official mlcroissant

Rad-R per-capture dashboard — healthy (snapshot t=39.0s, radar frame 390; all panels at this instant)

![](images/1dcad872551bf23eb81b21ba53fb350180f44c9de84e7581554f0efd90ba8fd2.jpg)

![](images/3fe0d012e72951244b9cdab0fc28be29e678daefdd1c7b37e872fd1566ed02d4.jpg)

![](images/0d3bf5d3e6ff90e171a606c77dddc5ee635d62854fe9c69be49b456bf51c10fe.jpg)

![](images/7ef3f1f656563b6e6af0894ba084329e04aa4fd132adf9d9a517ebc3bbe9f5a9.jpg)

![](images/a166eade2b539cf961f9cebab5e2cbccca6b604974ccc0a8245aa6fbc297f4f9.jpg)

![](images/9e11137473dd0de0c6fbe706c41e489072ad63438d065102c6262dde10e6e4c6.jpg)

t (s)  
![](images/527573d47ed40798756b39ca6306ecbe907d097bd502af19a25a781a442fb3f9.jpg)

![](images/7ce8644d6fb45c763b9dbe403bbd05abb9180713546ad3c293811a345d79d0be.jpg)

![](images/a14a16c977a3d18bff5a775013ced4b7b6ae2d5230269d8a8081555dee3746fe.jpg)  
t (s)  
Figure S5. Per-capture dashboard for healthy.

![](images/f3f40723fcdfbaa96363e51cacd998df9ded74480122df72e26ec2cf7127d2dd.jpg)  
t (s)

Rad-R per-capture dashboard — yaw0 (snapshot t=84.0s, radar frame 830; all panels at this instant)  
![](images/70b1e3438242f2fe9f91dac1f42ad74e40a8c98afb1b727c2b47b643dcb407fa.jpg)

![](images/826f0bd612379866e097fb1da79d3a274675953504437db54bd37e3324664271.jpg)

![](images/f740c9249aed5f5d0ce991f983aa8783ea36832f276114464cd6c55458609927.jpg)

![](images/d609fa1634b1d6dded5f81c8c4945cbf2f2ed1557acb480f9e25904a77899b1e.jpg)

![](images/96d71b2c37295cba22e7b33b01205fe5912461d259660a58c10dda099234a91e.jpg)

![](images/d950fed72ebf211131438f1dc702b3b1295179a657723344c21be65ca7475724.jpg)

t (s)  
![](images/ed10fefe90315471aa5c2f943e27c7479a302ba66666b206f0ea32306ca5db04.jpg)

![](images/cfb858e34bbcea4bacbfb0b434311d29391f1ededbcaa5f3c7a012b5627a698e.jpg)  
t (s)

![](images/72ef2d87a83f5034275a66f9cee0d2bd9d399d9beda1d48574af4622553a97e4.jpg)  
t (s)

![](images/29719c663e4445838cee48c708466a32412d3dfc1888c727fdb9421ebeacb681.jpg)  
Figure S6. Per-capture dashboard for $\mathtt { y a w O } ( 5 ^ { \circ }$ misalignment).  
t (s)

1.0 validator with no schema errors; two non-blocking @context warnings concern upstream extension keys.

## F. Hyperparameters and Training Setup

Table S2 summarizes the training hyperparameters used for every model in the benchmark. We use the same optimizer, schedule, and batch size across all models (epoch budget as noted below) so that the comparison isolates architecture;

t (s)  
![](images/1802eaa9a1a77c3d8beff387ff3e381858ddc68c48c79b63bd210ddd9f5ced59.jpg)

![](images/9ffbd77640a4627d54c5728aedf3f20af6e115df054f54d4469391887d43bde9.jpg)

![](images/0708907d927b11f36332312f10993671c2577de721db0673dfaab7cb2293039a.jpg)

![](images/f103b57529d27149b4da56366e9fe239da2df864e9cd1f30adfa199ff1fbad8b.jpg)

![](images/358d57c3d9fcd700e622f03a61b7eb4fefb64802dd4f057d112104874caa154d.jpg)

![](images/566efb93589117dd4edd54581a2c224cdc9af9be32e77345a51ade3ea4138248.jpg)

t (s)  
![](images/205151c52b340cc99dfcd0869e2a467956ce7249dd694a93e6ddaf2c49d901ea.jpg)

![](images/13d3bf5154dc0742ee0eaf71fac10b9e1b81bac00999c82ffce0a960a928f321.jpg)

![](images/3267d9716c6a02ebb8878a3c033339622d1e881cd2e65881d2e918fdbd893f79.jpg)  
t (s)

![](images/dddd7abc08e933e265313193c11260954a3cd16eae1ce81d465df4ea63b245f1.jpg)  
Figure S7. Per-capture dashboard for vib1 (low-amplitude mechanical vibration).

Rad-R per-capture dashboard — vib2 (snapshot t=218.0s, radar frame 2180; all panels at this instant)  
![](images/dd13456c7085f129a71ac383a3094c1c4dbe2a6b70fa1e9dfcc94da765092ddd.jpg)

![](images/91e204eb8c3f11d2a3ea199fbfe0a7428d36ab168456a8b90d0edbd406b2c4b9.jpg)

![](images/d670e3ce9f4935a815d4b38385acf52678815c9f35c873e69c64c835849aa878.jpg)

![](images/370ff654d0d8da565306799caafef09a7962b8bb4f28f593e78db37f41a94898.jpg)

![](images/b70bc526aa1153174991bacd9788c700f2a510dc910d9727f7ac9278e7652fcb.jpg)

![](images/4413f8e58b0c72124b85c16b080e6c9463559d6702b86fdf0c1033cf7702f862.jpg)

![](images/a72eb513e4643425808dd22e2c7a60851e62cc6262d06334bf029dca9930a054.jpg)  
t (s)

![](images/71f10dc3441ed4b4044fdb07bc476a1c6cd567c7fdab341fb1c9fd50ee1ac5af.jpg)  
t (s)

![](images/d430bc1294f6497cf409e4a840418b160d7d062f42c266842ae6acda876a09c2.jpg)  
t (s)

![](images/b207c8c8b1607bbeff455414ca102c8f2c0ea168157b283d2e544f08266cc0c0.jpg)  
t (s)  
Figure S8. Per-capture dashboard for vib2 (high-amplitude mechanical vibration).

per-model hyperparameter tuning would likely improve absolute numbers but obscure architectural conclusions.

Checkpoint-selection limitation. The saved within-clip and cross-severity runs do not use a separate validation partition: after each epoch, the script evaluates the designated test split and retains the checkpoint with maximum test fault macro-F1. Their reported values are therefore oracle-selected upper bounds. The few-shot protocol has no best-epoch selection and instead uses a fixed 30-step target adaptation schedule. We disclose this distinction because capture-disjoint splitting prevents frame overlap but does not remove checkpoint-selection bias.

Rad-R per-capture dashboard — blockage1 (snapshot t=200.0s, radar frame 1988; all panels at this instant)

![](images/295bb926cf7e492ffdfe13a692728ec61a6ff594e7f118d2c0830771c589b610.jpg)

![](images/ca94f15558f69db139d1b8443c11cb9347617fda93f55826072179763850506b.jpg)

![](images/4b411d98956eda3d674b9bb90d81907dc8804548fbc6c76a78cc18cf8f94d1d7.jpg)

![](images/df53cacee029fce366b161ccbe887e53bc77048bf74b72b7bfb7ff49503fbc99.jpg)

![](images/08abb78c8f8606098ba18b336d74569c3fb94fd5ca5621d9a758c2278cd566b5.jpg)

![](images/52914cfc2303c578fdf8ad18377191e10dfa58a5481ad976bcf4e95f8dda9048.jpg)

![](images/6599d3551421e16f9bf1a0b6efceb012a96c1afd9fed2e273b2f376172270c15.jpg)  
t (s)

![](images/dd326d0d7e4ed128978c3841f7f09b4786e5b4dffe0fdf91e229ae3b742f3262.jpg)

![](images/85395a0f96e4a8a433c299600297b8f9db3058b24c7710c77d8d0ba4217bf74a.jpg)  
t (s)

![](images/e12f042c2e8b24b85f7d0e023d8f01864df6c1a7ea14b9a7ec0c2898d09e3e52.jpg)  
Figure S9. Per-capture dashboard for blockage1 (S1: 30% radome coverage).

Rad-R per-capture dashboard — blockage2 (snapshot t=13.0s, radar frame 130; all panels at this instant)

![](images/f24cf529522a111f9b5ab7e9c4467f3782b68b1585245c2f8898b1dcddf88982.jpg)

![](images/baacd18483a773c0d2b76ffbeceb598a3bd6cfe4c9a680384077ae2f192651ae.jpg)

![](images/1afa5d20752d67ee8511e84b1e4c1cfa2e53c49f6f3985b58f7a818ebbcd901b.jpg)

![](images/92e6ebd58365b5821e60152923e70223738a76e9ef0823213991e68f0cd79538.jpg)

![](images/d36e293139d314ec292ad117af2a7cdd21cbf361e1730700e2fc0db83d0d049b.jpg)

![](images/7be2c85cc64ae4837d6681392fad43cb313aed80518499e15327ceb764591569.jpg)

![](images/137d375505824535d4140be87cf163ec77d408928f8a1565bf6d591ea961bcc0.jpg)  
t (s)

![](images/80cbce68131a9dc469c4bcdc26e9579e382099e46fbccdb8af177a8a4b5320e3.jpg)  
t (s)

![](images/14dce34392a05d3c78f204656b29960b0380b6966cc919f2d4cbc061e6fa6ace.jpg)  
t (s)

![](images/7dc0b081c02298f0fb47996d89be3280cdbd0a256fb06895f35af0c04efacdb3.jpg)  
t (s)  
Figure S10. Per-capture dashboard for blockage2 (S2: 60% radome coverage).

RadrNet hyperparameters. The default uses embedding dimension 256, stage depths [2, 2, 2], state size 16, convolution width 4, expansion 2, fast-axis patch size 8, and dropout 0.3. Training-only capture-invariance augmentations are a random global phase rotation $\theta \ : \sim \ : \mathcal { U } ( 0 , 2 \pi )$ per-Rx amplitude scaling $\sim \mathcal { U } ( 0 . 8 , 1 . 2 )$ , 5% Rx-channel dropout, and per-frame standardisation.

Rad-R per-capture dashboard — degrade0 (snapshot t=14.0s, radar frame 138; all panels at this instant)

![](images/8c2ff60dad2ea79a6e69d3dc87a0ffe31d1d285c11b6b20fcfd4cd9ff8244256.jpg)

![](images/001279a3a870c094b6f50101f0b0e2136cd70ae69bc3861e88708673d397e8dc.jpg)

![](images/3d3b9d6cc1e0bf7207acd132e7f2b914e4f7c85a2ad3a91141e6d91910fb49f7.jpg)

![](images/415c8d753362ed3c82b0364d833bd69fd8290a5a1010b3b0e2194bfccd9315d2.jpg)

![](images/26ff8aa6e3c9697d45f5e0da643a24d3423de662215ad43b71b0b8ce7575b11c.jpg)

![](images/6c6eb5a860a66800140f9bce6ba5855734f626ec76889eb1ae9dddabc2b6448c.jpg)

t (s)  
![](images/2f81f3df494d95fd6290e057a09b17b8d18cf0a3f8767de2cf93711b933ed50e.jpg)

![](images/9493d0cc7bf4de64c438097e1e1cfe21463cfde54fbebb4ecd5ae04b589d9f80.jpg)

![](images/a54acdc13252cf550690f72bfc7606a85b399aa02653a66f47f1e317d260d103.jpg)

![](images/4e9ae35beb190b780302eaa5a5afab692f35e24fa8e088d11b875ceaa8036761.jpg)  
Figure S11. Per-capture dashboard for degrade0 (S1: 6/16 Rx degraded).

Rad-R per-capture dashboard — degrade1 (snapshot t=243.0s, radar frame 2394; all panels at this instant)  
![](images/ef5ada6ecd2b57a80540ca741d5e19afd28fd1af9bed026cea4dc8c6c1f9edac.jpg)

![](images/853f219e1b57ddaccd0f626bc790a3e87e95c191c7ae1d3524d7842ebb5e7117.jpg)

![](images/061b145d64c8224b0169232f2a0264f7159601be7fadcee235c9a41236d92bed.jpg)

![](images/48cd17816e0bcbde4f6867e43f6ef6f67ae7fbba65a7c3730d3ec01fe2982868.jpg)

![](images/d0bfec9d9f75e68057e14ad03adcb80520df88762bf459164ff33fad65ba8fb6.jpg)

![](images/c97f585e278255fdb92fa6b32b260314dded3086916531ef62f3bb5a958775d4.jpg)

t (s)  
![](images/1df23788876ce9dd320613755751b89facaf6961cced84b9a314a24a3b0bf66c.jpg)

![](images/a798faf0493a8f494346892291efdee164da2c7c46929c1bb91a04294ad9147d.jpg)  
t (s)

![](images/7857012344d39db951482c9d5e7d762faef30ac40f68245cedcca5b326a1b215.jpg)  
t (s)

![](images/1132cee596a5956644864d754e880e64b9a028cff936a0e7380f4d984571a499.jpg)  
t (s)  
Figure S12. Per-capture dashboard for degrade1 (S2: 10/16 Rx degraded).

## G. Within-Clip Anytime Benchmark: Full Table

Table S3 gives the complete per-budget statistics for the within-clip chirp-wise anytime benchmark plotted in the main text (left panel of the two-axis figure): fault macro-

F1 mean±std over five seeds with BCa 95% bootstrap CIs, the one-sided paired-bootstrap p-value, and the McNemar two-sided exact p-value on 1 800 pooled samples.

Table S2. Training hyperparameters for the benchmark. Shared across models except epoch count and seeds (noted below); only the architecture differs.  
Optimizer AdamW   
Learning rate $1 \times 1 0 ^ { - 4 }$   
Weight decay $1 \times 1 0 ^ { - 4 }$   
LR schedule Cosine, $T _ { \mathrm { m a x } }$ = epoch budget   
Gradient-norm clip 1.0   
Batch size 16 (32 for single-Tx cache)   
Epochs 20 (cross-sev.: 20 or 25, per text)   
Loss CE(fault) + 0.5 CE(severity)   
Random seed fixed (multiple for multi-seed runs)   
Checkpoint selection max evaluation-split fault macro-F1 (oracle; see text)   
Hardware 1× RTX A4000 (16 GB)   
PyTorch 2.5.1+cu118   
Mamba kernels mamba ssm 2.3.1 (CUDA 11.8)

## H. Capture-Invariant SSM (RadrNet-CI / DS-CI)

The main paper introduces a capture-invariant SSM that reduces the cross-severity gap by encoding per-framestandardised magnitude and relative rather than absolute phase. This section specifies the encoding, reports the perfold breakdown, isolates the two encoding components and gives the per-fault recall of every model (Section H.4), provides the full anytime-under-cross-severity curves, and lists the hyperparameters.

## H.1. Relative-Phase Invariant Encoding

Let $x [ c , s , r ] ~ \in ~ \mathbb { C }$ be the raw IQ sample at chirp $c \in$ $\{ 0 , \ldots , C { - } 1 \}$ , ADC index s, and virtual channel r. The original RadrNet encoding stacks (Re x, Im x, |x|, ∠x) and therefore exposes the absolute phase $\angle x .$ Under a capture-specific transformation, each receiver experiences a constant phase offset and gain, $\begin{array} { r l } { x ^ { \prime } [ c , s , r ] } & { { } = } \end{array}$ $g _ { r } e ^ { j \left( \phi _ { 0 } + \psi _ { r } \right) } x [ c , s , r ]$ , where $\phi _ { 0 }$ is a global phase and $\psi _ { r }$ is a per-channel calibration phase. Because these nuisance terms are nearly constant within a recording but can differ between recordings, absolute phase can reveal capture identity and provide a shortcut under within-clip evaluation.

RadrNet-CI removes this by encoding two invariants. First, a per-frame standardised magnitude

$$
{ \tilde { m } } [ c , s , r ] = { \frac { | x [ c , s , r ] | - \mu } { \sigma } } ,
$$

where $\mu$ and $\sigma$ are the mean and standard deviation over $( c , s , r )$ . This removes a frame-wide offset and scale and is exactly invariant to global positive gain; it does not remove arbitrary per-channel gain changes. Second, the relative (slow-time) phase, the chirp-to-chirp difference

$$
\Delta \phi [ c , s , r ] = \angle x [ c + 1 , s , r ] - \angle x [ c , s , r ] ,
$$

encoded as (sin $\Delta \phi ,$ cos $\Delta \phi )$ to avoid wrap discontinuities. Differencing along c cancels any term constant in $c { \mathrm { : } }$ with $x ^ { \prime }$ as above, $\begin{array} { r } { \angle x ^ { \prime } [ c + 1 , s , r ] - \angle x ^ { \prime } [ c , s , r ] = \Delta \phi [ c , s , r ] } \end{array}$ , so both $\phi _ { 0 }$ and $\psi _ { r }$ vanish exactly. $\Delta \phi$ preserves the slow-time phase dynamics; which fault signatures each component actually carries is measured in Section H.4. The resulting input has $3 R$ channels $( \tilde { m } , \sin \Delta \phi , \cos \Delta \phi )$ versus the 4R of RadrNet; the invariance is a property of the architecture, not of a training-time augmentation. RadrNet-DS-CI feeds this invariant branch and a small RD-CNN branch (identical to RadrNet-DS’s) into the same 2-way input-dependent gate.

## H.2. Full Cross-Severity Ranking

Table S4 lists the per-model controlled cross-severity fault macro-F1 (three-seed mean±std) with input modality; these are the exact values behind the main-text ranking figure.

## H.3. Per-Fold Cross-Severity Breakdown

Figure S13 reports the two folds separately. Fold A trains on mild (S1) and tests on severe (S2); fold B reverses this. Fold B is the direction in which the off-the-shelf raw-IQ models collapsed: RadrNet-DS scores only 0.392 there. RadrNet-DS-CI recovers fold B to 0.678, the single largest source of its overall gain, while keeping fold A competitive; this is the quantitative form of the “fold-B fix” claimed in the main text.

## H.4. Encoding Ablation and Per-Fault Breakdown

To isolate the two components of the capture-invariant encoding, we trained RadrNet-CI with each component alone—per-frame-standardised magnitude only (R input channels) and relative phase only ((sin $\Delta \phi ,$ cos $\Delta \phi )$ , 2R channels)—keeping the backbone, augmentations, recipe, 2-fold cross-severity protocol and checkpoint rule identical to the full model (three seeds each). Table S5 reports fault macro-F1 with the per-fold means and, for every crossseverity model of the main text, the per-fault recall on the unseen test captures pooled over both folds and three seeds. Which component carries the gain. Per-frame magnitude standardisation is the dominant factor: on its own it raises the mean from 0.545 to 0.631 and recovers blockage recall from 23.4% to 67.3%. Relative phase on its own is insufficient (0.399, below the absolute-phase baseline) and does not raise the mean when added to magnitude (0.612 vs. 0.631, within the seed spread). Its contribution is confined to the harder severe→mild fold B $( 0 . 5 3 8 \to 0 . 5 7 8 )$ , where it lifts blockage $( 7 6 . 8 \%  8 9 . 3 \% )$ and Rx-degradation $( 7 2 . 7 \% \to 9 1 . 8 \% )$ recall, at a cost in fold A $( 0 . 7 2 4  0 . 6 4 7$ , mainly misalignment, 93.7% → 66.3%). Relative phase alone recovers vibration poorly (30.8%), so the chirp-to-chirp phase difference is not where the vibration signature is most readily read off. The invariance property of Section H.1 (exact cancellation of constant global and per-channel phase offsets) is unaffected, but the cross-severity gain of RadrNet-CI over RadrNet should be attributed mainly to magnitude standardisation. Without checkpoint selection (fixed 20-epoch schedule, last epoch), the full-frame fault macro-F1 is 0.497 (magnitude only), 0.382 (relative phase only) and 0.580 (both).

Table S3. Chirp-wise anytime benchmark on the within-clip 80/20 MIMO split (five seeds, 20 epochs/seed). Each cell is fault macro-F1, mean±std across seeds with the BCa 95% bootstrap CI in brackets. $p _ { \mathrm { b o o t } }$ is the one-sided paired-bootstrap p-value (resampling floor $5 \times 1 0 ^ { - 4 } ) ;$ p<sub>McN</sub> is the McNemar two-sided exact p-value on 1 800 pooled samples. Numbers verified against results/v1 anytime summary.json.
<table><tr><td rowspan="2">k</td><td colspan="2">F1 mean ± std [BCa 95% CI]</td><td rowspan="2">∆F1</td><td rowspan="2">Pboot</td><td rowspan="2">PMcN</td></tr><tr><td>ResNet-18</td><td>RadrNet-DS</td></tr><tr><td>4</td><td> $0 . 1 5 3 \pm 0 . 1 0 4 [ . 0 7 7 , . 2 3 2 ]$ </td><td> $\mathbf { 0 . 7 4 7 \pm 0 . 0 6 8 [ . 6 7 3 , . 7 8 7 ] }$ </td><td>+0.594</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>8</td><td> $0 . 1 2 8 \pm 0 . 0 7 1 [ . 0 7 7 , . 1 8 4 ]$ </td><td> $\mathbf { 0 . 7 6 1 \pm 0 . 0 4 7 \ [ . 7 2 8 , . 8 0 3 ] }$ </td><td>+0.633</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>16</td><td> $0 . 0 9 0 \pm 0 . 0 2 0 [ . 0 7 5 , . 1 0 6 ]$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 3 4 } \ [ . 7 4 6 , . 8 0 0 ]$ </td><td>+0.682</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>32</td><td> $0 . 1 2 1 \pm 0 . 0 4 2 [ . 0 9 3 , . 1 5 8 ]$ </td><td>0.804 ± 0.043 [.774, .843]</td><td>+0.683</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>48</td><td> $0 . 4 0 2 \pm 0 . 0 8 3 [ . 3 3 9 , . 4 6 5 ]$ </td><td> $\mathbf { 0 . 9 3 6 \pm 0 . 0 2 3 [ . 9 1 6 , . 9 5 4 } ]$ </td><td>+0.534</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $< 1 0 ^ { - 4 }$ </td></tr><tr><td>64</td><td> $0 . 9 2 6 \pm 0 . 0 1 7 [ . 9 1 2 , . 9 3 9 ]$ </td><td> $\mathbf { 0 . 9 6 1 \pm 0 . 0 1 9 } \ [ . 9 4 7 , . 9 7 6 ]$ </td><td>+0.035</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $5 . 9 \times 1 0 ^ { - 5 }$ </td></tr></table>

![](images/f17febbacf231590f6ccf38b33505484fb2f6c5df0af8c2a3fe32705152e64e8.jpg)  
Figure S13. Per-fold controlled cross-severity fault macro-F1 (three-seed means): Fold A (train S1, test S2) vs. Fold B (train S2, test S1) for each model.

Why the RD branch hurts the absolute-phase model but helps the invariant one. With absolute phase, adding the RD branch collapses fold-B Rx-degradation recall from 98.8% to 17.3%, which is the whole drop from 0.545 to 0.485; with the invariant encoding, the RD branch lifts fold-B vibration $( 3 1 . 5 \% \to \ 8 5 . 5 \% )$ and misalignment $( 4 0 . 8 \%  6 4 . 0 \% )$ recall at a modest blockage cost $( 8 9 . 3 \%  7 7 . 7 \% )$ . Every model recovers every fault on captures it never saw during training (the only cell near chance is absolute-phase blockage in fold B), so the models separate faults rather than captures; capture-specific cues

instead hurt transfer.

## H.5. Anytime Under Cross-Severity

Table S6 gives the anytime curves under the cross-severity split. RadrNet-DS-CI is substantially weaker at partial budgets and reaches its highest prefix-sweep score only at k=64, consistent with the RD branch benefiting from a better-resolved Doppler spectrum. RadrNet-CI reaches 0.479 fault macro-F1 from 4 of 64 chirps (three-seed mean) and increases monotonically, giving the strongest reported small-budget performance under cross-severity at the cost of a lower full-frame score than DS-CI. ResNet-18 shows a similar dependence on chirp budget. The absolute-phase RadrNet-DS remains below RadrNet-CI throughout and trails ResNet-18 at the full frame (0.441 vs. 0.591). The

Table S4. Controlled cross-severity 2-fold benchmark, fault macro-F1 (mean±std over three seeds). From results/v1 2fold <sub>\*</sub> seed<sub>\*</sub>.json.
<table><tr><td>Model</td><td>Input</td><td>F1</td></tr><tr><td>RadrNet-DS-CI</td><td> $\mathrm { R D } + \mathrm { i n v . } \mathrm { I Q }$ </td><td> $\mathbf { 0 . 6 6 3 \pm 0 . 0 0 9 }$ </td></tr><tr><td>ResNet-18</td><td>RD map</td><td> $0 . 6 2 8 \pm 0 . 0 1 5$ </td></tr><tr><td>RadrNet-CI</td><td>inv. IQ</td><td> $0 . 6 1 2 \pm 0 . 0 1 5$ </td></tr><tr><td>DeiT-III</td><td>RD map</td><td> $0 . 5 8 1 \pm 0 . 0 2 4$ </td></tr><tr><td>Swin-T</td><td>RD map</td><td> $0 . 5 6 8 \pm 0 . 0 1 0$ </td></tr><tr><td>EfficientNetV2</td><td>RD map</td><td> $0 . 5 6 6 \pm 0 . 0 1 3$ </td></tr><tr><td>RadrNet</td><td>raw IQ</td><td> $0 . 5 4 5 \pm 0 . 0 0 6$ </td></tr><tr><td>PVTv2</td><td>RD map</td><td> $0 . 5 3 8 \pm 0 . 0 0 6$ </td></tr><tr><td>ConvNeXt-V2</td><td>RD map</td><td> $0 . 5 2 9 \pm 0 . 0 1 8$ </td></tr><tr><td>RadrNet-DS</td><td>RD + raw IQ</td><td> $0 . 4 8 5 \pm 0 . 0 4 1$ </td></tr><tr><td>ViT-Small</td><td>RD map</td><td> $0 . 4 8 0 \pm 0 . 0 1 0$ </td></tr></table>

DS-CI k=64 value (0.601) is produced by the separate prefix-evaluation path and flat averaging over six seed×fold runs; the headline 0.663 instead averages fold-level fullframe metrics within each seed and then across seeds. The endpoints are therefore not directly interchangeable. These results show a trade-off between early raw-IQ evidence and full-frame RD evidence; they do not isolate a single causal component.

## H.6. Few-Shot Label-Budget Axis

Table S7 and Figure S14 report all five models under the identical few-shot protocol of the main text (15-epoch base training, 30 adaptation steps at lr $1 0 ^ { - 4 }$ , 2 folds × 3 support draws per seed). RadrNet-DS and ResNet-18 use three seeds; DeiT-III, RadrNet-CI, and RadrNet-DS-CI use one. RadrNet-DS-CI places second overall, above every published baseline from k=5 (e.g. 0.732 vs. DeiT-III’s 0.711 at k=5), while the absolute-phase RadrNet-DS remains the most label-efficient (0.755/0.780/0.785 at $k { = } 5 / 1 0 / 2 0 )$ The invariant encoding thus trades a small amount of fewshot adaptability for its cross-severity generalisation and anytime robustness, completing the trade-space picture: no single variant wins every axis.

## H.7. Hyperparameters

RadrNet-CI and RadrNet-DS-CI reuse RadrNet’s backbone and training recipe exactly: embed dim=256, depths=[2,2,2], d state=16, d conv=4, expand=2, patch size fast=8, dropout=0.3; AdamW at lr $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , cosine schedule, gradient-norm clip 1.0, batch size 16, 20 epochs, oracle checkpoint selection by evaluation-split fault macro-F1, and loss $\mathrm { C E } ( \mathrm { f a u l t } ) + 0 . 5 \mathrm { C E } ( \mathrm { s e v e r i t y } )$ . The only architectural change is the input encoding: the invariant branch ingests the 3R-channel ( ˜m, sin $\Delta \phi , \cos \Delta \phi )$ tensor of Section H.1 in place of RadrNet’s 4R-channel real/imag/mag/abs-phase tensor; the absolute-phase augmentations are consequently unnecessary for the invariant branch. Parameter counts are ∼4.1 M (RadrNet-CI) and ∼4.7 M (RadrNet-DS-CI), both smaller than RadrNet-DS (∼5.1 M). The cross-severity benchmark uses the 2-fold train-one-severity/test-the-other protocol of the main text; both CI and DS-CI are run for three seeds.

![](images/1cc89b1bf9b263aff3391c86bf1229e702a5e13307d2d40b7b8c6ed7e8bbd415.jpg)  
Figure S14. Few-shot label-budget curve: fault macro-F1 vs. labelled frames per class k for the five compared models. RadrNet-DS and ResNet-18 show three-seed aggregates; the other curves are single-seed. Error bars and exact values are in Table S7.

## I. Cross-Modal Comparison: Radar Micro-Doppler vs. IMU

We compare a hand-computed radar feature with the independently measured IMU signal. For each frame of the healthy, vib1, and vib2 captures (200 frames per capture), we compute the off-centre Doppler energy $1 - p _ { 0 }$ where $p _ { 0 }$ is the zero-Doppler bin’s share of the slow-time power spectrum averaged over fast time and 192 virtual channels. IMU vibration energy is the RMS of the demeaned accelerometer-plus-gyroscope magnitude over the corresponding frame window. Pooled over frames, the descriptive rank association is Spearman $\rho ~ = ~ 0 . 4 1 $ : conditions with greater IMU vibration also show broader radar micro-Doppler. Within either vibration capture, however, the frame-wise association is approximately zero. Because the 600 frames come from only three condition-level captures and are temporally correlated, we do not interpret a frame-wise permutation test as capture-level significance; the result supports between-condition ordering rather than within-capture tracking.

## J. Per-Fault Error Analysis under Cross-Severity Shift

To show where the cross-severity gap comes from, Figure S16 reports row-normalized confusion matrices for the capture-invariant headline model (RadrNet-DS-CI) and the range–Doppler CNN baseline (ResNet-18), pooled over all three seeds and both fold directions at the full chirp budget $( k { = } 6 4 , n { = } 4 8 0 0$ test predictions per model). The four induced faults are the true classes (healthy has no severity to cross), but the five-way classifiers can still predict healthy, so that column is shown separately. Vibration is recovered well by both models (86–89%). The dominant failure mode is collapse to healthy: a third of misalignment frames read as healthy for the invariant model (66% recall, vs. 85% for the RD-CNN), and over a third of blockage frames for both, consistent with these faults being subtle in the averaged power map. Blockage is the hardest class (39% recall for the invariant model, 30% for the RD-CNN), and the RD-CNN additionally confuses it with receive degradation (29%); both recover receive degradation well (86%). The blockage- and misalignment-tohealthy confusion is the principal headroom in the benchmark. These matrices are computed from the final-epoch weights of the prefix-evaluation path (Section H.5), whereas the per-fault recalls in Table S5 come from the best-epoch records behind Table S4; the latter are therefore higher (e.g. 56.8% vs. 39% blockage recall for RadrNet-DS-CI), and the two views should not be mixed.

Table S5. Encoding ablation and per-fault breakdown under the controlled cross-severity 2-fold protocol (three seeds). F1: fault macro-F1 mean±std over seeds; Fold A/B: per-fold means (A: train S1, test S2; B: train S2, test S1); Vib./Misal./Block./Rx-deg.: recall (%) of each true fault on the unseen test captures. Encoding: abs. = real/imag/magnitude/absolute phase (4R); mag. = standardised magnitude only (R); rel. = relative phase only (2R); both = RadrNet-CI encoding (3R). From results/v1 2fold .json.
<table><tr><td>Model</td><td>Encoding, input</td><td>F1</td><td>Fold A</td><td>Fold B</td><td>Vib.</td><td>Misal.</td><td>Block.</td><td>Rx-deg.</td></tr><tr><td>RadrNet</td><td>abs., IQ</td><td> $0 . 5 4 5 \pm 0 . 0 0 6$ </td><td>0.655</td><td>0.434</td><td>84.7</td><td>77.2</td><td>23.4</td><td>94.4</td></tr><tr><td>RadrNet-DS</td><td>abs., IQ+RD</td><td> $0 . 4 8 5 \pm 0 . 0 4 1$ </td><td>0.578</td><td>0.392</td><td>89.2</td><td>74.9</td><td>26.2</td><td>54.2</td></tr><tr><td>RadrNet-CI</td><td>mag., IQ</td><td> $0 . 6 3 1 \pm 0 . 0 2 8$ </td><td>0.724</td><td>0.538</td><td>73.0</td><td>67.0</td><td>67.3</td><td>86.3</td></tr><tr><td>RadrNet-CI</td><td>rel., IQ</td><td> $0 . 3 9 9 \pm 0 . 0 2 5$ </td><td>0.388</td><td>0.410</td><td>30.8</td><td>43.4</td><td>49.2</td><td>67.7</td></tr><tr><td>RadrNet-CI</td><td>both, IQ</td><td> $0 . 6 1 2 \pm 0 . 0 1 5$ </td><td>0.647</td><td>0.578</td><td>63.0</td><td>53.6</td><td>69.8</td><td>95.8</td></tr><tr><td>RadrNet-DS-CI</td><td>both, IQ+RD</td><td> $\mathbf { 0 . 6 6 3 \pm 0 . 0 0 9 }$ </td><td>0.648</td><td>0.678</td><td>91.7</td><td>72.5</td><td>56.8</td><td>94.7</td></tr><tr><td>ResNet-18</td><td>RD map</td><td> $0 . 6 2 8 \pm 0 . 0 1 5$ </td><td>0.675</td><td>0.582</td><td>87.3</td><td>83.7</td><td>54.2</td><td>77.1</td></tr></table>

Table S6. Anytime-under-cross-severity fault macro-F1 at chirp budget k. RadrNet-CI and RadrNet-DS-CI are three-seed means over seeds×folds; ResNet-18 and the absolute-phase RadrNet-DS are a single seed (mean over the 2 folds). From the anytime curve fields of v1 2fold {dsci,ci,anytime} seed<sub>\*</sub>.json.
<table><tr><td>k</td><td>RadrNet-CI</td><td>RadrNet-DS-CI</td><td>ResNet-18</td><td>RadrNet-DS</td></tr><tr><td>4</td><td>0.479</td><td>0.304</td><td>0.127</td><td>0.257</td></tr><tr><td>8</td><td>0.488</td><td>0.308</td><td>0.128</td><td>0.273</td></tr><tr><td>16</td><td>0.504</td><td>0.310</td><td>0.142</td><td>0.283</td></tr><tr><td>32</td><td>0.546</td><td>0.319</td><td>0.146</td><td>0.287</td></tr><tr><td>48</td><td>0.571</td><td>0.332</td><td>0.181</td><td>0.295</td></tr><tr><td>64</td><td>0.580</td><td>0.601</td><td>0.591</td><td>0.441</td></tr></table>

Table S7. Few-shot cross-capture adaptation (label-budget axis), fault macro-F1 at label budgets $k \geq 1 .$ RadrNet-DS and ResNet-18 are aggregated over three seeds × two folds × three support draws; the other models use one seed × two folds × three draws.
<table><tr><td>k</td><td>ResNet-18</td><td>DeiT-III</td><td>RadrNet-DS</td><td>RadrNet-CI</td><td>RadrNet-DS-CI</td></tr><tr><td>1</td><td> $0 . 5 7 4 \pm 0 . 0 7 3$ </td><td> $0 . 5 0 6 \pm 0 . 0 9 3$ </td><td> $\mathbf { 0 . 5 8 4 \pm 0 . 0 7 3 }$ </td><td> $0 . 4 9 1 \pm 0 . 1 3 1$ </td><td> $0 . 5 6 1 \pm 0 . 0 7 7$ </td></tr><tr><td>5</td><td> $0 . 5 6 2 \pm 0 . 0 9 1$ </td><td> $0 . 7 1 1 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 7 5 5 \pm 0 . 0 4 0 }$ </td><td> $0 . 6 7 5 \pm 0 . 0 6 0$ </td><td> $0 . 7 3 2 \pm 0 . 0 4 3$ </td></tr><tr><td>10</td><td> $0 . 6 5 3 \pm 0 . 1 0 4$ </td><td> $0 . 7 4 4 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 7 8 0 \pm 0 . 0 1 6 }$ </td><td> $0 . 7 1 9 \pm 0 . 0 5 0$ </td><td> $0 . 7 5 7 \pm 0 . 0 1 2$ </td></tr><tr><td>20</td><td>0.685 ± 0.071</td><td> $0 . 7 3 8 \pm 0 . 0 4 1$ </td><td> $\mathbf { 0 . 7 8 5 \pm 0 . 0 1 4 }$ </td><td> $0 . 7 5 4 \pm 0 . 0 2 3$ </td><td> $0 . 7 6 1 \pm 0 . 0 1 8$ </td></tr></table>

![](images/c061bd359613651a861fe53453bb00f909e36b57ff88b7eb178c2ca1c03896a9.jpg)  
Figure S15. Descriptive cross-modal comparison. Radar offcentre Doppler energy vs. independently measured IMU vibration energy, one point per frame, pooled over the healthy and two vibration captures (Spearman $\rho = 0 . 4 1 )$ . Frames are temporally correlated within only three condition-level captures; the plot therefore supports a between-condition trend, not a capture-level significance claim.

## K. Additional Analyses

Accuracy–efficiency trade-off. Figure S17 plots controlled cross-severity fault macro-F1 against parameter count. The capture-invariant models are Pareto-optimal: RadrNet-DS-CI attains the top score (0.663) at 4.7 M parameters, under half the size of ResNet-18 (11 M) and roughly a fifth of the transformer backbones (20–28 M), all of which it outperforms. RadrNet-CI is the smallest model overall (4.1 M) yet still beats every RD-map backbone except ResNet-18. Parameter count alone therefore does not explain the cross-severity ranking.

Within-clip vs. cross-severity: a re-ranking. Figure S18 contrasts each model’s within-clip and controlled crossseverity fault macro-F1. Within clip, most models nearsaturate (0.87–0.98; the weakest, ViT-Small, is at 0.71), with the raw-IQ models among the leaders. Under crossseverity shift, every model declines and the absolute-phase raw-IQ models fall furthest (RadrNet-DS 0.977 → 0.485), dropping below the RD-map CNNs they previously led. This re-ranking is consistent with capture-specific phase and gain acting as within-clip shortcuts. The captureinvariant models recover the top of the cross-severity ranking.

![](images/0218bde649ce6292e6e668f27e7d008d00272914081e26b308e57bdb7f88e658.jpg)  
Figure S16. Cross-severity per-fault confusion. Rownormalized recall (%) for RadrNet-DS-CI (left) and ResNet-18 (right), pooled over three seeds and both fold directions at the full chirp budget (k=64). The true classes are the four induced faults (healthy has no severity to cross); the separated Healthy column counts frames the five-way model misreads as healthy.

![](images/3e306a39cbccb68fa284580ec5bc009026c4d8e7805ab3ce66fa5b294779bdee.jpg)  
Figure S17. Accuracy–efficiency trade-off. Controlled crossseverity fault macro-F1 (three-seed mean) vs. model size (parameters, log scale). Marker colour denotes model family (captureinvariant, absolute-phase raw-IQ, RD-map backbone); the dashed line is the Pareto frontier.

## L. Compute Resources

The benchmark was run end-to-end on a single Windows-11 workstation with two NVIDIA RTX A4000 GPUs (16 GB each), accessed via WSL2 / Ubuntu 22.04. Only one GPU is used at a time. Approximate compute budget:

• One training cache build (raw .bin → HDF5 with RD maps + IQ): ∼11 min for single-Tx, ∼13 min for full 192- channel MIMO cache.

• Per-model within-clip MIMO training: ∼5 min for ResNet-18, ∼7 min for the transformer backbones (ViT-Small, DeiT-III, Swin-T, PVTv2), ∼5–7 min for ConvNeXt-V2 / EfficientNetV2, and ∼25 min for the RadrNet family (the dominant cost is the Mamba selective scan with checkpointing).

• Anytime benchmark: five seeds × 2 models (ResNet-18, RadrNet-DS) at 20 epochs/seed, ∼5 GPU-hours; the chirp-budget evaluation sweep itself is inference-only.

• Few-shot benchmark: 15-epoch base training plus 30- step adaptation per support draw across the label budgets, <2 GPU-hours.

• Controlled cross-severity benchmark (2 folds per run; 25 epochs for ConvNeXt-V2, DeiT-III, Swin-T, PVTv2, EfficientNetV2 and the single-stream RadrNet, 20 for ResNet-18, ViT-Small, RadrNet-DS and the captureinvariant runs): three seeds for all 11 models in the reported ranking, plus the prefix-budget evaluations and CI few-shot run, for ∼10 GPU-hours.

• Total project compute including iteration, ablations, the published baselines, and the capture-invariant crossseverity runs: <25 GPU-hours.

One command reproduces the within-clip benchmark: pythonscripts / train \_ within \_ clip \_ mimo . py -- cache . . . /training \_ cache . h5. The scripts aggregate\_anytime.py and aggregate\_ fewshot.py aggregate the two budget studies. Individual jobs complete within an hour on an A4000; the full multiseed anytime sweep takes approximately 5 GPU-hours.

## M. Capture Protocol Details (Lab Playbook Excerpts)

The code repository provides the full capture protocol, bill of materials, severity-gate measurements, per-clip metadata schema, and printable checklist (docs / FAULT \_ INDUCTION\_SOP.md; docs/FAULT\_INDUCTION\_ CHECKLIST.pdf). Table S8 summarises the benchmark settings.

Capture session protocol. Each session: (1) power on and thermally stabilise for 10 min; (2) record a healthy baseline; (3) induce faults in a session-seeded random order; (4) for each fault, measure the severity gate, record 5 min, remove the fixture, and record a 5 s recovery reference; (5) record a closing healthy reference; and (6) run per-file SHA-256 and automated QA via src/io/qa.py. Randomisation reduces temporal-position confounding.

## N. Future Directions

Future releases aim to broaden Rad-R along several axes: additional severity levels, multi-day / multi-device / multiscene capture, further fault and interference modalities, and additional model baselines. Concrete specifications will accompany those releases.

��������������������� $\neq$ ��������������������������  
![](images/06ee6797fafb594fcc1eb18dbab4c39c65548da283c166ba94b6f9ee42b99100.jpg)  
Figure S18. Within-clip vs. cross-severity macro-F1 per model. Each point is a model at its within-clip fault macro-F1 (x, from v1 within clip mimo<sub>\*</sub>.json) and three-seed cross-severity F1 (y, the main cross-severity table values). Marker colour denotes raw-IQ (orange) vs. RD-map (grey); the shaded band marks the capture-invariant models.

## O. Broader Impact

Positive. Better automotive-radar fault diagnosis could support earlier detection of sensor degradation in ADAS and autonomous-driving systems. Releasing raw ADC under an open license lowers the barrier to academic radarperception research, historically constrained by proprietary data and expensive hardware. The fault-induction SOP and printable checklist standardise the capture protocol so future researchers can extend Rad-R reproducibly.

Negative / risks. (a) Camera privacy: the two co-recorded camera streams during outdoor captures may incidentally capture pedestrians or license plates. Every released camera frame is processed through an automated PII-redaction pipeline that Gaussian-blurs every detected human face and licence plate before upload; the unredacted originals never leave the authors’ workstation, and a 5%-per-capture random sample is manually verified for missed detections. (b) Adversarial misuse: a fault classifier could in principle be inverted into a fault-induction recipe. We judge this risk low: the released fault-induction procedures are already public, the underlying physics is well known to anyone with a 77 GHz radar, and adversarial induction requires physical access to the vehicle. We do not release productiondeployment detector weights; the released weights are research checkpoints only. (c) Over-claiming generalisation: practitioners might read these numbers as production-ready. We mitigate this by treating within-clip F1 as an upper bound, framing the dataset as a single-day / single-device research dataset, and committing to broaden coverage in future releases (Appendix N).

Table S8. Per-fault induction details for the captures.
<table><tr><td>Fault</td><td>Hardware</td><td>S1 (mild)</td><td>S2 (severe)</td></tr><tr><td>Vibration</td><td>Eccentric-mass DC tor, bolted to radar bracket, quency BNO055 IMU records ground- truth tri-axial acceleration</td><td></td><td>mo- 20 Hz drive fre- 40 Hz drive fre- quency</td></tr><tr><td>Misalign.</td><td>Custom precision yaw-offset 5° yaw jig, 1° graduations, locked with set-screw, verified with  $0 . 1 ^ { \circ }$  digital inclinometer</td><td></td><td>10° yaw</td></tr><tr><td>Blockage</td><td>1/16-inch clear polycarbonate 30% coverage of 60% coverage sheet on top of the always- the radar aperture installed 1/16-inch polypropy- lene radome; laser-cut coverage templates, glossy face inward</td><td></td><td></td></tr><tr><td>tion</td><td>Rx degrada- 3M 1181 copper foil tape (0.5- 6 of 16 Rx chan- 10 of 16 Rx chan- inch wide, 25 µm thick) fully nels covered covering targeted Rx antenna patches; per-channel dB drop verified by single-tone calibra- tion sweep before each capture</td><td></td><td>nels covered</td></tr></table>

## P. Reproducibility and Responsible-AI Summary

This section consolidates the reproducibility and responsible-AI details supporting the main-text claims.

Reproducibility. All models share the same core optimizer, learning rate, weight decay, batch size, cosine schedule, and gradient clipping; protocol-specific epoch budgets and seed counts are stated in Appendix F. The appendix also records the RadrNet configuration and capture-invariance augmentations. The within-clip split definition, cache-build script, training/evaluation scripts, and result JSONs are included in the release so that each reported protocol can be reconstructed from its stated configuration.

Data and code access. The code release includes training and evaluation code, result JSONs, Croissant 1.0 metadata, and a compact PII-redacted data sample. The complete dataset and code will be released publicly; data will be licensed under CC BY 4.0 and code under the licenses stated above. The metadata fields cover data collection, annotation protocol, maintenance, personal/sensitive information, limitations, biases, use cases, social impact, sources, provenance, and preprocessing.

Statistical significance. The anytime benchmark reports five-seed mean±std with BCa-95% bootstrap CIs, onesided paired-bootstrap p-values (n=2000) and McNemar exact p-values on 1 800 pooled samples; the few-shot benchmark uses a one-sided unpaired bootstrap (n=2000) over the support draws, with RadrNet-DS and ResNet-18 carried across three seeds and DeiT-III reported at a single seed.

Compute. The full benchmark runs on two NVIDIA RTX A4000 GPUs (16 GB; one used at a time) under WSL2 / Ubuntu 22.04, CUDA 11.8, PyTorch 2.5.1, mamba ssm 2.3.1. Per-model within-clip wall-clock: ∼5 min ResNet-18, ∼7 min the transformer backbones, ∼25 min RadrNet; the anytime benchmark adds ∼5 GPU-hours (five seeds, 2 models) and the few-shot benchmark <2 GPU-hours; total project compute <25 GPU-hours; cache build ∼11–13 min. Research ethics. No human-subject, medical, or deliberately collected sensitive personal data. Incidental pedestrians / plates in outdoor camera frames are redacted as above. All hardware was operated on private lots and permitted campus roads at low speed; no public-road testing of faultinduced sensors occurred. Fault-induction fixtures use commercial off-the-shelf hardware and are fully reversible.