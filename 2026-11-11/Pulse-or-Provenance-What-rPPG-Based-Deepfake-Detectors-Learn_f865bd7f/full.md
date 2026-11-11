# Pulse or Provenance? What rPPG-Based Deepfake Detectors Learn

Othmane Harraq

Temple University

othmane.harraq@temple.edu

Tamer Aldwairi Temple University aldwairi@temple.edu

Abstract— Remote photoplethysmography (rPPG) has been proposed as a physiological cue for deepfake detection, based on the premise that synthesized faces lack a genuine cardiac signal. We examine what rPPG-style detectors learn on Celeb-DF++ (22 forgery methods) and FaceForensics++ (FF++), using one identically sampled window per real and fake video. On talkingface forgeries, simple chrominance traces reach an AUC of 0.998, exceeding a dedicated rPPG extractor (0.87). Controlled experiments indicate that this signal is not physiological in origin. On fully synthesized forgeries, it is equally discriminative when extracted from background regions or from full frames without face detection, and it is concentrated above the 0.7 to 4 Hz heart-rate band. It is largely removed by shuffling frame order but persists after normalizing resolution, frame rate and encoding, and an independent reimplementation reproduces these findings. Removing high-frequency content makes most synthesized videos appear real, adding it makes real videos appear fake, and injecting a synthetic pulse has little effect. In controlled self-reenactment experiments with two generators, animating a face from a single frame introduces the same highfrequency signal, whereas reconstructing each frame from its own source does not. Face-based chrominance detectors trained on Celeb-DF++ classify over 99% of real FF++ videos as fake, indicating that they largely encode video source rather than manipulation, and a CLIP-based detector exhibits the same behavior. A frozen CLIP backbone matches or exceeds fusions of CLIP with rPPG and frequency features. Finally, we show that the official Celeb-DF and Celeb-DF++ splits share identities and source-video content between training and test, and that custom protocols reusing test videos and identities in training inflate weaker detectors by up to 0.30 AUC.

## I. INTRODUCTION

Physiological signals are an attractive basis for deepfake detection because they come with a mechanistic argument. Remote photoplethysmography (rPPG) recovers the small, periodic color changes that blood flow produces in facial skin [1], [2], [3], and a synthesized face has no blood flow. Detectors built on this premise, including FakeCatcher [5], DeepRhythm [6], and DeepFakesON-Phys [7], report high accuracy on face-swap benchmarks. The same reasoning underlies pulse-based face anti-spoofing [9], [10].

The premise is strongest for talking-face (TF) forgeries, which animate a single still image from audio and contain no real video from which a pulse could be inherited. Face-swap forgeries, by contrast, can carry the pulse of their source footage [14]. TF forgeries are therefore the setting in which an rPPG detector should work best and for the reasons its designers intend.

On the TF subset of Celeb-DF++ [25], the result is puzzling. A plain chrominance (CHROM) trace [2] separates real from TF videos with an AUC of 0.998 (Fig. 1), while a state-of-the-art rPPG extractor, RhythmFormer [4], reaches only 0.87. If the discriminative information were a heartbeat, a cruder pulse estimate should not outperform a better one.

![](images/101ea7e934a4f15c459643b1803c2cf78ab8c9e5df2cab048d0087211b63bef7.jpg)

![](images/5ccca585641b76a33674691821f1d6ce4c7e191889b81553b24f965230376e12.jpg)  
Fig. 1. rPPG-style detectors do not need a face or a heartbeat. Top: real vs. fake AUC by region and signal. Bottom: face CHROM with frames shuffled or restricted to one frequency band. On talking-face (TF) and facereenactment (FR) forgeries, the background and the full frame separate real from fake as well as the face, the 4–12 Hz band beats the 0.7–4 Hz heartrate band, and shuffling frames removes the signal. Matched sampling, test identities; dashed line is chance.

Prior work asked whether forgeries carry a valid pulse [13], [14]; we ask what the detectors use instead and where it comes from. Under a matched sampling protocol, in which every real and fake video contributes one identically positioned window, we test where the signal lives, what kind of signal it is, whether it survives format normalization, how it responds to targeted interventions, and what it says about the video it is computed from. Our findings are:

• The signal is not primarily physiological. On fully synthesized forgeries it is as strong in background regions and in full frames without face detection as on the face. It is concentrated above the 0.7 to 4 Hz heartrate band, disappears when frame order is shuffled, and survives normalization of resolution, frame rate, and encoding. An independent reimplementation reproduces these results.

• It is introduced by image-to-video synthesis. In controlled self-reenactment experiments with two generators, animating a face from a single frame of a real video adds the same high-frequency signal, while reconstructing each frame from itself does not.

• The detectors largely encode video source. Facebased chrominance detectors trained on Celeb-DF++ label over 99% of real FF++ videos as fake, and the same features separate two collections of real videos. A CLIP-based detector shows the same behavior.

• Evaluation practice obscures this. The official Celeb-DF and Celeb-DF++ splits share identities and source content between training and test, and custom protocols that reuse test videos inflate weak detectors by up to 0.30 AUC.

We close with a short set of controls that physiological detectors should report. Code and evaluation splits will be made publicly available.

## II. RELATED WORK

Physiological signals for liveness and deepfake detection. rPPG estimation ranges from classical color-space projections such as CHROM [2] and POS [3] to learned extractors such as RhythmFormer [4]. Pulse cues were first used for face anti-spoofing, where a printed photo or a replayed screen should show no pulse [9], [10], [15], and later for deepfake detection [5], [6], [7], [8]. Harraq and Aldwairi [11] applied rPPG-derived signals to TF forgeries but did not establish that the discriminative content was cardiac, and later combined these signals with lip-region DCT features [12]. Two studies question the premise directly. D’Amelio et al. [13] found that rPPG-based detection on face-swap data does not track signal quality as expected, and Seibold et al. [14] showed that face-swap forgeries can inherit a valid pulse from their driver video. We study the opposite case, forgeries with no driver video, and identify what the detectors use instead.

Temporal artifacts in video forgeries. A separate line of work detects forgeries from temporal inconsistency without appealing to physiology, for example through mouth motion [16], short-range temporal coherence [17], or inconsistencies across multiple time scales [18], [19]. Our findings place rPPG-style detectors in this group: the signal they use is a temporal color artifact, measured through a physiological lens.

Shortcuts and dataset bias. Classifiers often exploit features that are predictive in a dataset but unrelated to the intended task [20]. In deepfake detection, a leading silence in fake audio was shown to separate real and fake samples in two widely used audio-visual datasets [21], compressionformat differences bias generated-image detectors [22], and binary detectors absorb identity information [23]. Our contribution is to show that a detector class justified by biology is largely a source and synthesis detector, and to measure this with controls specific to temporal signals.

Benchmarks and splits. Celeb-DF [24] and its extension Celeb-DF++ [25] provide 59 celebrities and, in Celeb-DF++, 22 forgery methods across face-swap (FS), face-reenactment (FR), and TF categories, with an official test list. FF++ [26] defines splits by pairs of source videos. We examine how these splits separate identities and content.

## III. EXPERIMENTAL SETUP

## A. Data and splits

We use all 22 methods of Celeb-DF++ [25] and the five manipulation methods of FF++ (c23) [26]. For Celeb-DF++, we partition the 59 identities into 41 training, 9 validation, and 9 test identities. FS and FR filenames name two identities; a fake is assigned to a split only if both identities belong to it, and is excluded otherwise. TF filenames name one celebrity. For FF++ we use the official split. Section VII additionally analyzes the official Celeb-DF and Celeb-DF++ test lists. All reported results use test identities only.

## B. Matched temporal sampling

Real and fake videos differ in length and frame rate. To prevent the sampling procedure from becoming a cue, every video, real or fake, contributes exactly one window: its first 160 frames. Videos shorter than 160 native frames are excluded rather than resampled; this removes all Hyper-Reenact videos, so that method has no matched result. The length is the training chunk length of the RhythmFormer UBFC configuration and was inherited, not tuned. On TF, face CHROM reaches 0.998 at 80, 128, and 160 frames, the band, shuffle, and background results hold at each length, and selection on validation identities also picks 160 (Section IV). Real Celeb-DF++ videos are 30 fps, while most generators output 25 fps; frequency analyses therefore use the fps25 variant, in which real videos are downsampled to 25 fps by frame dropping and both classes are re-encoded identically.

## C. Signals and regions

For each window we compute per-frame mean RGB values over three regions. The face region is one MediaPipe BlazeFace [27] box per video, enlarged 1.5× and fixed for all frames. The background region is a square of the same size at the frame corner farthest from the face, if it does not overlap the enlarged box; otherwise, as in fake videos whose face fills the frame, it is the mean over all pixels outside the box. The full-frame region is the whole image, with no face detection. From each RGB series we derive three traces. The green trace is the green channel. CHROM [2] normalizes each channel by its temporal mean, forms $X = 3 R - 2 G$ and $Y \ = \ 1 . 5 R + G - 1 . 5 B ,$ and returns X − αY with $\alpha = \sigma ( X ) / \sigma ( Y )$ . POS [3] projects the normalized channels onto the plane orthogonal to skin tone and combines the two projections with a standard-deviation ratio. We also run RhythmFormer [4] (UBFC checkpoint) on the face crops. Each trace is z-scored per window before classification.

Frequency bands. To isolate frequency ranges, we zero the real FFT coefficients of the raw window outside a band, at the video’s frame rate, and then z-score the result. The bands are below 0.7 Hz, 0.7–4 Hz (the usual heart-rate range), and 4–12 Hz, capped below the 12.5 Hz Nyquist frequency of 25 fps video so that both classes are filtered identically.

## D. Classifiers

Traces are classified with a small 1D ResNet with three residual stages and dropout 0.5, trained for a fixed 30 epochs with AdamW (learning rate $1 0 ^ { - 3 } .$ , weight decay $5 \times 1 0 ^ { - 4 }$ batch 64) and a class-weighted binary cross-entropy loss. As general-purpose references we use small MLP heads on frozen CLIP ViT-L/14 [28] and Xception [29] features, averaged over 10 frames of the same window (frames 0, 16, . . . , 144), and fusions of CLIP with rPPG and lip-region DCT features [12]. The rPPG+DCT detector concatenates MLP embeddings of the face RhythmFormer waveform and a 64-coefficient DCT descriptor of a 64×64 lip patch (23K parameters); the fusion projects CLIP, rPPG, and DCT features to 256 dimensions each and combines them with learned branch weights. Validation identities are used only for early stopping of these heads; the trace classifiers do not use them.

## E. Evaluation

We report video-level AUC as the mean over 5 seeds with 95% bootstrap confidence intervals (1,000 resamples of test videos), and identity-level bootstrap intervals for headline results. Because real test videos are outnumbered by fakes (up to 1:6), we also report precision-recall metrics: for TF face CHROM, PR-AUC is 0.998 with fakes as the positive class and 0.999 with reals as the positive class, the EER is 1.0%, and AUC on class-balanced subsamples is 0.998. For TF RhythmFormer they are 0.770, 0.914, 19.7%, and 0.869, so weaker detectors look worse under these metrics than under AUC. Face detection fails for at most 1% of Celeb-DF++ videos and 2–3% of FF++ videos in either class; the 160-frame rule excludes 43% of TF fakes and 9% of FS and FR fakes but almost no real videos. Random-label controls are reported as the distribution over 20 label draws. All runs are deterministic. An independent reimplementation that shares no code with the main pipeline reproduces the main results (Section IV).

## IV. THE SIGNAL IS NOT PRIMARILY PHYSIOLOGICAL

## A. Near-perfect detection from a crude signal

On TF forgeries, a CHROM trace from the face separates real and fake videos with an AUC of 0.998 [0.997, 1.000] (Table I). POS performs identically. RhythmFormer, a dedicated rPPG extractor, reaches 0.869 [0.827, 0.907], and a green-channel trace 0.931. The ordering is the reverse of what a physiological explanation predicts: the signals that are worst at estimating a pulse are best at detection. The identity-level bootstrap interval for CHROM, [0.996, 1.000], is as narrow as the video-level one.

## B. The face is not needed

On fully synthesized forgeries the signal does not depend on skin. A background region gives 0.998 on TF and 0.998 on FR, and the mean color of the full frame, computed without any face detection, gives 0.999 and 1.000 (Table I). Even the green channel is stronger from the background than from the face (0.960 vs. 0.931 on TF, 0.915 vs. 0.614 on FR). Face-detector instability is not the cause: our traces use one box per video, and box jitter measured per frame is lower in fake videos than in real ones. On FS and FF++, which keep real footage outside the face, the background is much weaker than the face (0.720 vs. 0.888 and 0.537 vs. 0.694). We return to this difference in Section V.

## C. The signal is temporal and lies above the heart-rate band

Shuffling the frames within each window removes the signal: CHROM falls to 0.521 on TF and 0.505 on FR, and to chance on FS and FF++. Static color statistics of the same windows carry no information (single-feature AUC 0.48– 0.55). Among simple per-window statistics, only temporal ones separate the classes: fake traces vary less over the window as a whole (temporal standard deviation, AUC 0.25) but are less smooth from frame to frame (AUC 0.905 for a smoothness score). The discriminative content is therefore in how color changes from frame to frame.

To locate it in frequency, we filter each window into three bands before normalization. On TF, the 4–12 Hz band alone reaches 0.992, the 0.7–4 Hz heart-rate band 0.943, and the band below 0.7 Hz 0.644. The high band exceeds the heartrate band in every set we tested, for CHROM and POS, and at every window length (80, 128, 160 frames). The power spectra agree (Fig. 2): on TF, 17.1% [16.8, 17.5] of the face CHROM trace power lies in 4–12 Hz for fakes against 12.1% [11.4, 12.7] for reals, and on FR the gap is larger (19.4% vs. 6.3%). On FS the excess is small (8.4% vs. 6.3%), and on FF++ it is absent (13.1% vs. 13.4%). RhythmFormer is the exception, as expected for a model trained to output pulseband waveforms.

## D. The signal is not a file-format artifact

In the native files, real and TF videos differ in codec, frame rate, keyframe interval, resolution, and audio. Re-encoding both classes to the same frame rate and codec, cropping and resizing faces to a common resolution, and combining all of these leaves the AUC at 0.986 or higher (Table II); after full normalization no metadata field separates the classes. Notching out the frequencies of the real videos’ original keyframe interval changes the AUC no more than an equally wide control notch (0.995 vs. 0.999). Heavier compression weakens but does not remove the cue: at CRF 40, models trained on standard encodes reach 0.857 [0.833, 0.882], and 0.931 [0.913, 0.947] after retraining. A standalone reimplementation, using a different classifier and no shared code, reproduces every conclusion.

0.7-4 Hz heart-rate band  
Frequency (Hz)  
TABLE I  
REAL VS. FAKE AUC UNDER MATCHED SAMPLING (ONE WINDOW PER VIDEO, TEST IDENTITIES, MEAN OF 5 SEEDS). SHUFFLED AND BAND COLUMNS USE THE FACE SIGNAL. BANDS ARE APPLIED BEFORE PER-WINDOW NORMALIZATION. TF USES THE FPS25 VARIANT. COUNTS ARE FAKE/REAL TEST VIDEOS. RHYTHMFORMER OPERATES ONLY ON DETECTED FACE CROPS, SO IT HAS NO BACKGROUND OR FULL-FRAME ENTRIES (–).
<table><tr><td>Set</td><td>Signal</td><td>Face</td><td>Background</td><td>Full frame</td><td>Shuffled</td><td>0.7-4Hz</td><td>4-12Hz</td></tr><tr><td rowspan="4">TF (384/98)</td><td>CHROM</td><td>0.998</td><td>0.998</td><td>0.999</td><td>0.521</td><td>0.943</td><td>0.992</td></tr><tr><td>POS</td><td>0.998</td><td>0.993</td><td>0.998</td><td>0.593</td><td>0.897</td><td>0.970</td></tr><tr><td>Green</td><td>0.931</td><td>0.960</td><td>0.959</td><td>0.604</td><td>0.802</td><td>0.911</td></tr><tr><td>RhythmFormer</td><td>0.869</td><td></td><td></td><td>0.679</td><td>0.860</td><td>0.800</td></tr><tr><td rowspan="4">FR (315/99)</td><td>CHROM</td><td>1.000</td><td>0.998</td><td>1.000</td><td>0.505</td><td>0.809</td><td>0.996</td></tr><tr><td>POS</td><td>1.000</td><td>0.995</td><td>1.000</td><td>0.493</td><td>0.740</td><td>0.994</td></tr><tr><td>Green</td><td>0.614</td><td>0.915</td><td>0.853</td><td>0.551</td><td>0.602</td><td>0.672</td></tr><tr><td>RhythmFormer</td><td>0.701</td><td></td><td>1</td><td>0.484</td><td>0.662</td><td>0.695</td></tr><tr><td rowspan="4">FS (568/99)</td><td>CHROM</td><td>0.888</td><td>0.720</td><td>0.860</td><td>0.496</td><td>0.547</td><td>0.798</td></tr><tr><td>POS</td><td>0.877</td><td>0.750</td><td>0.895</td><td>0.482</td><td>0.507</td><td>0.790</td></tr><tr><td>Green</td><td>0.568</td><td>0.660</td><td>0.609</td><td>0.515</td><td>0.519</td><td>0.518</td></tr><tr><td>RhythmFormer</td><td>0.509</td><td></td><td></td><td>0.465</td><td>0.523</td><td>0.503</td></tr><tr><td rowspan="4">FF++ (207/139)</td><td>CHROM</td><td>0.694</td><td>0.537</td><td>0.623</td><td>0.504</td><td>0.500</td><td>0.637</td></tr><tr><td>POS</td><td>0.684</td><td>0.555</td><td>0.620</td><td>0.527</td><td>0.558</td><td>0.645</td></tr><tr><td>Green</td><td>0.629</td><td>0.492</td><td>0.535</td><td>0.512</td><td>0.540</td><td>0.587</td></tr><tr><td>RhythmFormer</td><td>0.549</td><td>一</td><td></td><td>0.506</td><td>0.567</td><td>0.573</td></tr></table>

![](images/ec5dbd1ea219e4f0041cdb11ad544a0ef95439dec882189a3cf9312b9b2905d0.jpg)

![](images/0360df7151cf0d3b5c4e70404d69cb4baddeb0a7c59fcbc99ab05a5052d3a727.jpg)

![](images/9b3727bad6902fa7c6fc389d3246cab83d7a9c668e3fbacc1eb13a034747041e.jpg)

![](images/c221c66ae3963e94ae0e1fd8fef60925f903c74747275d022a8bb079373e79ef.jpg)

![](images/3d8b1368c79a6470174758f2c98c1eb61fc2ab1e5b64e23ab69755bb2e8dd94f.jpg)

![](images/b3460737f36ef34271a8ad7786b3cceefcfd8a60ac90f06a727cf93329889f63.jpg)

![](images/e2a7c67504af8cf056f52357e645b0bbf2e85fc64790a9d52610fe21c77f16cf.jpg)

![](images/aca93f96e3a7059fe80c2c707e56f4153e05fa3de57a69d5b4048fe61bb3909b.jpg)

![](images/f7077803b5b9c0dc4445d5d9162038ec24a7e28060b22fb544c616115fc01fa7.jpg)

![](images/49f4a9165698682394e6d55cd342cd14c25cb1cdbc655d5b9e1fd6f0b874a21e.jpg)

![](images/7c0390a111506397619a35272565655e98004abc57bd49e1d4692247656da53a.jpg)

![](images/db66f0809955b3d8dc41a0e38ada9284e989be9ec104304b8e57d9f4b8465cf6.jpg)  
Fig. 2. Averaged power spectra of real (black) and fake (blue) CHROM traces under matched sampling. Rows: face, background, full frame. Columns: forgery category. The shaded band is 0.7–4 Hz. Real and fake spectra separate above 4 Hz for TF and FR, only slightly for FS, and not at all for FF++.

## E. Interventions

Removing high-frequency content from fake traces makes nearly all of them appear real: after low-passing at 8 Hz,

97% of fakes are called real, against 2.0% without modification (Table III). The models rely most on content nearest the 12.5 Hz Nyquist limit, that is, on alternation between

## TABLE II

TF FACE CHROM UNDER PROGRESSIVELY STRONGER NORMALIZATION (MATCHED SAMPLING), AND AN INDEPENDENT REIMPLEMENTATION (LOGISTIC REGRESSION ON THE TRACE SPECTRUM, NO SHARED CODE). BAND COLUMNS ARE N/A FOR NATIVE FILES, WHOSE FRAME RATES DIFFER BETWEEN CLASSES (25 VS. 30 FPS), AND FOR SHUFFLED TRACES, WHICH HAVE NO TEMPORAL STRUCTURE TO FILTER.
<table><tr><td>Variant</td><td>Plain</td><td>0.7-4 Hz</td><td>4-12Hz</td></tr><tr><td>Native files</td><td>0.998</td><td>N/A</td><td>N/A</td></tr><tr><td>fps25 (frame rate, codec matched)</td><td>0.998</td><td>0.943</td><td>0.992</td></tr><tr><td>res_norm (face crop 2562, 30 fps)</td><td>0.998</td><td>0.859</td><td>0.993</td></tr><tr><td>res25 (crop, resolution, fps, codec)</td><td>0.986</td><td>0.906</td><td>0.978</td></tr><tr><td>Reimplementation, face</td><td>0.989</td><td>0.802</td><td>0.983</td></tr><tr><td>Reimplementation, background</td><td>0.995</td><td>0.698</td><td>0.952</td></tr><tr><td>Reimplementation, shuffled</td><td>0.531</td><td>N/A</td><td>N/A</td></tr></table>

TABLE III

INTERVENTIONS ON TEST TRACES WITH UNMODIFIED TF FACE-CHROM MODELS (FPS25, MATCHED SAMPLING). FLIPPED: FRACTION OF THE MODIFIED CLASS ON THE WRONG SIDE OF THE UNMODIFIED MODELS’ EER THRESHOLD. LOW-PASS FILTERS ALSO REMOVE CONTENT UP TO THE 12.5 HZ NYQUIST LIMIT.
<table><tr><td>Intervention</td><td>AUC</td><td>Flipped</td></tr><tr><td>None</td><td>0.998</td><td>2.0% of each class</td></tr><tr><td>Fakes low-passed at 8 Hz</td><td>0.589</td><td>97% of fakes</td></tr><tr><td>Fakes low-passed at 4 Hz</td><td>0.877</td><td>85% of fakes</td></tr><tr><td>Fakes, 12–12.5 Hz removed</td><td>0.825</td><td>75% of fakes</td></tr><tr><td>Fakes, 4–12 Hz removed</td><td>0.999</td><td>1.6% of fakes</td></tr><tr><td>Reals + 4–12 Hz noise (fake level)</td><td>0.998</td><td>1% of reals</td></tr><tr><td>Reals + 12–12.5 Hz noise (fake level)</td><td>0.510</td><td>99.8% of reals</td></tr><tr><td>Fakes + synthetic pulse</td><td>0.996</td><td>2.5% of fakes</td></tr><tr><td>Reals, 0.7–4 Hz removed</td><td>0.996</td><td>3.7% of reals</td></tr><tr><td>Reals, 4.5–7.8 Hz removed (control)</td><td>0.990</td><td>9.0% of reals</td></tr><tr><td>Reals, 7.8–11.1 Hz removed (control)</td><td>0.990</td><td>15.7% of reals</td></tr></table>

consecutive frames. Removing only 12–12.5 Hz makes 75% of fakes appear real, while removing 4–12 Hz leaves them detected (1.6%), because the near-Nyquist content that remains is enough. Low-passing at 4 Hz flips fewer fakes than at 8 Hz because it also makes them smoother than real traces (mean absolute frame-to-frame change 0.253 vs. 0.426), which the models treat as atypical. This content is also sufficient. TF fakes carry 26% of their trace power in 12–12.5 Hz against 1.4% for reals; raising real traces to the fake level in this band makes 99.8% of them appear fake (AUC 0.510), whereas fake-level noise spread over 4–12 Hz does not (1%). Injecting a synthetic pulse into fake traces, a 1.0–1.7 Hz sinusoid with the mean heart-rate-band power of real traces, leaves 2.5% [1.7, 3.4] of them called real, close to the 2.0% baseline, and doubling its power gives 6.2%. The presence of a cardiac component is therefore not what makes a video look real to these models. Removing 0.7–4 Hz from fake traces changes nothing (2.2% called real). Conversely, removing the 0.7– 4 Hz band from real traces flips only 3.7% [1.6, 6.3] of reals, fewer than removing an equally wide band outside the heart-rate range (9.0% for 4.5–7.8 Hz, 15.7% for 7.8– 11.1 Hz). The chrominance detectors are therefore no more sensitive to the heart-rate band than to arbitrary bands of the same width. RhythmFormer is the exception: removing the heart-rate band flips 13.7% of reals against 5.1% and 1.4% for the control bands, consistent with an extractor trained to output pulse-band waveforms. It is also the weakest of the four detectors (Table I).

## V. WHERE THE SIGNAL COMES FROM

## A. Controlled self-reenactment

The previous section shows what the signal is not. To test whether synthesis introduces it, we regenerate 199 real Celeb-DF videos from themselves with two Celeb-DF++ reenactment methods, LivePortrait [31] and the Thin-Plate Spline Motion Model (TPSMM) [30]. Each reenactment shares its person, scene, motion, and final encoding with its source video. In mode (a) the face is animated from the first frame and pasted into it; in mode (b) the full face crop is generated; in mode (a2) each frame is reconstructed from itself.

Animating a face from a single frame makes the reenactment separable from its own source video, under the same matched sampling as before (Fig. 3). For LivePortrait, face CHROM reaches 0.830 [0.744, 0.902] in mode (a) and 0.931 [0.878, 0.971] in mode (b); for TPSMM, 0.767 [0.656, 0.865] and 0.896 [0.835, 0.944]. Both show the signature of the Celeb-DF++ forgeries: the 4–12 Hz band exceeds the heart-rate band (LivePortrait 0.711 vs. 0.539 and 0.818 vs. 0.607; TPSMM 0.728 vs. 0.497 and 0.793 vs. 0.652), the share of high-band power rises above that of the source video (LivePortrait 16.7% and 17.8% vs. about 10%; TPSMM 12.4% and 14.3%), and shuffling removes the effect. Reenactments also carry more power just below the Nyquist limit than their sources (at 30 fps, 14.5–15 Hz: LivePortrait 10.0% and 11.3% vs. about 3.5%; TPSMM 5.7% and 6.3%), though far less than TF fakes do. The effect is weaker for TPSMM than for LivePortrait. Reconstructing each frame from itself with LivePortrait adds almost nothing (0.582 [0.465, 0.707]; high-band power 10.2% vs. 10.1%); TPSMM has no frame-by-frame mode, so this control uses LivePortrait only. The cue is thus introduced by synthesizing a sequence from a still appearance, not by the rendering and pasting pipeline. In mode (a), the frozen background is also separable, which reflects the absence of motion rather than added content, and we do not count it as evidence. The test split contains 32 to 34 clips per class, so intervals are wide.

## B. Consistency across forgery types

This account predicts the pattern across forgery types in Table I, and it holds method by method (Table IV). TF and FR methods generate the whole frame from a still source and are detected almost perfectly, including from the background. FS methods replace the face in real footage and are detected less well, with a much weaker background. The two FS methods that largely escape are the original Celeb-DF forgeries (0.604), whose generation pipeline was designed to reduce temporal flicker [25], and MobileFaceSwap (0.721). FF++ manipulations edit only the face of real footage; under matched sampling, detection there is weak (0.694), the background is near chance, and fakes carry no excess high-band power. Sampling matters here: when real videos contribute windows from across their full length while fakes contribute their first window, FF++ detection rises to 0.901, and a size-matched control shows that the increase comes from comparing later real windows with the first fake window, not from the amount of training data. RhythmFormer, which outputs pulse-band waveforms, is near chance on FS and FF++.

![](images/9cce7a4087efbc4890075d55882546c11296b11a3fc0f00e5a7434e06e478560.jpg)

![](images/a926370e7eb185a758a1fa1eab2cccddaf0ab6d38915c85a80a069aa5753661e.jpg)  
Fig. 3. Real videos vs. their own self-reenactments. Top: face CHROM AUC for separating each reenactment from its source video, plain, shuffled, and per frequency band. Bottom: share of face CHROM power in 4–12 Hz for each real video and its reenactments. (a) animated from the first frame and pasted back; (b) generated face crop; (a2) each frame rebuilt from itself. LP: LivePortrait.

## VI. THE DETECTORS ENCODE VIDEO SOURCE

## A. Scoring real videos from another source

If a detector learned a property of forgeries, real videos from a different dataset should be scored as real. Detectors trained on Celeb-DF++ real and fake videos instead label 99% or more of real FF++ videos as fake (Table V, Fig. 4), and their fake score separates held-out Celeb-DF real videos from FF++ real videos almost perfectly. YouTube-real videos, collected through the same pipeline as Celeb-DF real videos, are mostly scored as real by the CHROM detectors.

TABLE IV  
PER-METHOD REAL VS. FAKE AUC UNDER MATCHED SAMPLING (MEAN OF 5 SEEDS; TEST IDENTITIES; ABOUT 98–139 REAL TEST VIDEOS PER SET). FACE, BG AND FRAME ARE CHROM TRACES FROM THE FACE, BACKGROUND AND FULL FRAME; RF IS RHYTHMFORMER. <sup>†</sup>ECHOMIMIC OUTPUTS 24 FPS AND IS SCORED ON NATIVE FILES. CELEB-DF DENOTES THE ORIGINAL CELEB-DF-V2 FORGERIES.
<table><tr><td></td><td>Method</td><td> $n _ { \mathrm { f a k e } }$ </td><td>Face</td><td> $\mathbf { B g }$ </td><td>Frame</td><td>RF</td></tr><tr><td>FS</td><td>BlendFace</td><td>55</td><td>0.945</td><td>0.772</td><td>0.911</td><td>0.534</td></tr><tr><td></td><td>Celeb-DF</td><td>183</td><td>0.604</td><td>0.509</td><td>0.629</td><td>0.473</td></tr><tr><td></td><td>GHOST</td><td>55</td><td>0.973</td><td>0.756</td><td>0.923</td><td>0.502</td></tr><tr><td></td><td>HifiFace</td><td>55</td><td>0.953</td><td>0.769</td><td>0.907</td><td>0.539</td></tr><tr><td></td><td>InSwapper</td><td>55</td><td>0.954</td><td>0.749</td><td>0.905</td><td>0.482</td></tr><tr><td></td><td>MobileFaceSwap</td><td>55</td><td>0.721</td><td>0.507</td><td>0.695</td><td>0.490</td></tr><tr><td></td><td>SimSwap</td><td>55</td><td>1.000</td><td>0.950</td><td>0.998</td><td>0.528</td></tr><tr><td></td><td>UniFace</td><td>55</td><td>0.951</td><td>0.747</td><td>0.909</td><td>0.524</td></tr><tr><td>FR</td><td>DaGAN</td><td>52</td><td>1.000</td><td>0.999</td><td>1.000</td><td>0.690</td></tr><tr><td></td><td>FSRT</td><td>52</td><td>1.000</td><td>0.998</td><td>1.000</td><td>0.724</td></tr><tr><td></td><td>HyperReenact</td><td></td><td>no video ≥160 frames</td><td></td><td></td><td></td></tr><tr><td></td><td>LÍA</td><td>52</td><td>1.000</td><td>0.999</td><td>1.000</td><td>0.724</td></tr><tr><td></td><td>LivePortrait</td><td>55</td><td>1.000</td><td>0.997</td><td>1.000</td><td>0.672</td></tr><tr><td></td><td>MCNET</td><td>52</td><td>1.000</td><td>0.998</td><td>1.000</td><td>0.682</td></tr><tr><td></td><td>TPSMM</td><td>52</td><td>1.000</td><td>0.999</td><td>1.000</td><td>0.713</td></tr><tr><td>TF</td><td>AniTalker</td><td>68</td><td>1.000</td><td>0.998</td><td>1.000</td><td>0.905</td></tr><tr><td></td><td>EchoMimic†</td><td>50</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.786</td></tr><tr><td></td><td>EDTalk</td><td>68</td><td>1.000</td><td>0.998</td><td>1.000</td><td>0.898</td></tr><tr><td></td><td>FLOAT</td><td>64</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.821</td></tr><tr><td></td><td>IP-LAP</td><td>58</td><td>0.990</td><td>0.993</td><td>0.996</td><td>0.711</td></tr><tr><td></td><td>Real3D-Portrait</td><td>65</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.953</td></tr><tr><td></td><td>SadTalker</td><td>61</td><td>1.000</td><td>0.999</td><td>1.000</td><td>0.926</td></tr><tr><td>FF++</td><td>Deepfakes</td><td>42</td><td>0.607</td><td>0.524</td><td>0.574</td><td>0.662</td></tr><tr><td></td><td>Face2Face</td><td>41</td><td>0.694</td><td>0.561</td><td>0.640</td><td>0.435</td></tr><tr><td></td><td>FaceSwap</td><td>42</td><td>0.696</td><td>0.495</td><td>0.609</td><td>0.592</td></tr><tr><td></td><td>FaceShifter</td><td>41</td><td>0.729</td><td>0.540</td><td>0.653</td><td>0.596</td></tr><tr><td></td><td>NeuralTextures</td><td>41</td><td>0.746</td><td>0.564</td><td>0.642</td><td>0.459</td></tr></table>

TABLE V

REAL VIDEOS FROM OTHER SOURCES, SCORED BY DETECTORS TRAINED ON CELEB-DF++ (MATCHED SAMPLING). CALLED FAKE: FRACTION ABOVE EACH MODEL’S OWN EER THRESHOLD. LAST COLUMN: AUC SEPARATING HELD-OUT CELEB-REAL FROM FF++ REAL USING THE FAKE SCORE.
<table><tr><td>Training</td><td>Model</td><td>YouTube</td><td>FF++</td><td>Celeb vs FF++</td></tr><tr><td>TF</td><td>Face CHROM</td><td>1%</td><td>99.1%</td><td>0.998</td></tr><tr><td>TF</td><td>Background CHROM</td><td>3%</td><td>77.8%</td><td>0.959</td></tr><tr><td>TF</td><td>CLIP</td><td>38%</td><td>86.0%</td><td>0.948</td></tr><tr><td>All 22</td><td>Face CHROM</td><td>18%</td><td>99.7%</td><td>0.999</td></tr><tr><td>All 22</td><td>CLIP</td><td>47%</td><td>87.8%</td><td>0.945</td></tr></table>

In practical terms, these detectors largely learned that a video not from the Celeb-DF real collection is fake. This is not a generic shift between datasets, since YouTube-real videos, which share the Celeb-DF collection pipeline but not its exact videos, are mostly scored as real. POS-based detectors behave the same way (94–100% of FF++ reals called fake).

## B. Real versus real

The same features separate two collections of real videos when trained to do so: Celeb-DF real vs. FF++ real reaches 1.000 for CHROM and for CLIP, and Celeb-DF real vs.

![](images/73e73064c4c21beff2abfe8dc5778c62b7010e2f9ba1e10e7833fe10f29153c7.jpg)

![](images/032aae17962a027f3c1d8f4ba3391ada38358baeec8d26cc8797e822b8b54bdc.jpg)  
Fig. 4. The detectors encode video source. Top: fraction of each real collection called fake by detectors trained on the TF subset or on all 22 Celeb-DF++ methods, at each detector’s own EER threshold. Bottom: classifiers trained to separate two collections of real videos, with a randomlabel control.

YouTube-real reaches 0.727 for CHROM and 0.968 for CLIP. With random labels, the classifiers stay at chance (CHROM 0.515 ± 0.050, CLIP 0.505 ± 0.063 over 20 label draws), so this is not a tendency to find differences everywhere.

## C. General-purpose detectors

Source sensitivity is not limited to temporal traces. A frozen CLIP backbone also labels most FF++ real videos as fake (Table V). On a common protocol, CLIP and face CHROM are comparable overall (0.957 vs. 0.951, with overlapping 95% CIs), with face CHROM stronger on TF and FR and weaker on FS (Table VI). Adding rPPG and frequency features to CLIP does not help. On the strict identity-disjoint protocol the fusion scores 0.938 against 0.957 for CLIP alone, and under the benchmark’s GF-eval protocol 0.880 against 0.885; across eight protocols, fusion never exceeds CLIP alone by more than 0.004. Removing the rPPG or DCT branch from the trained fusion leaves its AUC unchanged, while removing CLIP collapses it (0.555 on GF-eval). For reference, DeepFakesON-Phys [7] reaches 0.575 on our TF test set, and 0.548 with our face crops, so its weakness is not due to its face detector.

## TABLE VI

TOP: DETECTORS ON THE SAME STRICT IDENTITY-DISJOINT PROTOCOL, MATCHED SAMPLING, 95 REAL AND 1,301 FAKE TEST VIDEOS SHARED BY ALL FOUR DETECTORS, MEAN OF 5 SEEDS. BOTTOM: CLIP VS. FUSION ON THE MATCHED FEATURE CACHE, WITH 95% CIS; STRICT USES A FIXED 30 EPOCHS, AS IN TABLE VIII. GF-EVAL FOLLOWS THE BENCHMARK PROTOCOL, IN WHICH ABOUT 90% OF TEST REALS ARE ALSO USED IN FINE-TUNING; WITH ONLY THE OFFICIAL TEST REALS, THE THREE ROWS GIVE 0.844, 0.830, AND 0.513.
<table><tr><td>Model</td><td>FS</td><td>FR</td><td>TF</td><td>All</td></tr><tr><td>Face CHROM</td><td>0.877</td><td>1.000</td><td>0.992</td><td>0.951</td></tr><tr><td>CLIP (frozen)</td><td>0.971</td><td>0.956</td><td>0.943</td><td>0.957</td></tr><tr><td>Xception (frozen)</td><td>0.673</td><td>0.841</td><td>0.806</td><td>0.765</td></tr><tr><td>RhythmFormer</td><td>0.499</td><td>0.634</td><td>0.727</td><td>0.614</td></tr><tr><td></td><td colspan="2">Strict</td><td colspan="2">GF-eval</td></tr><tr><td>CLIP only</td><td></td><td>0.957 [0.946, 0.968]</td><td colspan="2">0.885 [0.877, 0.891]</td></tr><tr><td>CLIP + rPPG + DCT</td><td></td><td>0.938 [0.923, 0.953]</td><td colspan="2">0.880 [0.872, 0.887]</td></tr><tr><td> $\mathrm { r P P G } + \mathrm { D C T }$ </td><td></td><td>0.680 [0.633, 0.726]</td><td colspan="2">0.547 [0.534, 0.560]</td></tr></table>

TABLE VII

OVERLAP BETWEEN TRAINING AND TEST IN THE OFFICIAL SPLITS (TEST SIDE). <sup>∗</sup>FF++ HAS NO IDENTITY LABELS; ITS SPLIT IS DEFINED BY SOURCE VIDEOS. <sup>†</sup>NO TF SUBSET.
<table><tr><td></td><td>Celeb-DF++</td><td>Celeb-DF-v2</td><td>FF++</td></tr><tr><td>Test fakes sharing an identity</td><td>100%</td><td>100%</td><td>_*</td></tr><tr><td>FS/FR test fakes sharing a pair</td><td>100%</td><td>100%</td><td>0%</td></tr><tr><td>FS/FR test fakes on a training real</td><td>78-80%</td><td>69%</td><td>0%</td></tr><tr><td>TF test fakes from a training real</td><td>85%</td><td>N/A†</td><td>N/A†</td></tr><tr><td>Test reals used for training fakes</td><td>92-100%</td><td>90%</td><td>0%</td></tr></table>

VII. EVALUATION PITFALLS ON THE CELEB-DF FAMILY

## A. Overlap in the official splits

The official Celeb-DF and Celeb-DF++ test lists are defined per video, not per identity (Table VII). Every test fake shares an identity with training data, every FS and FR test fake shares its identity pair with training fakes, and most test fakes were generated from a real video in the training set. Every Celeb-DF++ test real was used as the source of a training TF fake. These overlaps are content, not only identity: for 99.7% of test fakes, the training video they were made from is more similar in CLIP feature space than the average of the same person’s other training videos. The FF++ split, defined by pairs of source videos, has no such overlap.

Whether the overlap changes reported results depends on the model and on how the overlapping item relates to the test item. We remove overlapping training items and compare against removing the same number of random items. Training on the official Celeb-DF++ training list without FF++ pretraining, removing training fakes that share an identity pair with test fakes gives CLIP 0.759, against 0.819 for random removal (paired difference −0.061 [−0.085, −0.036]), suggesting that this same-label overlap inflates CLIP by about 0.06. Removing training reals that served as sources for test fakes, a cross-label overlap, raises AUC relative to random removal, slightly for CLIP (+0.012) and more for a frozen Xception head (+0.040), so this kind of overlap deflates results. The official split therefore does not bias all detectors in the same direction, which makes comparisons between detectors on it harder to interpret.

TABLE VIII  
LEAVE-ONE-METHOD-OUT (LOMO) AND IDENTITY-DISJOINT PROTOCOLS ON CELEB-DF++, MATCHED SAMPLING. LOOSE: IDENTITIES AND REAL VIDEOS SHARED. CLEAN: OFFICIAL TEST REALS HELD OUT. STRICT: IDENTITIES HELD OUT. LOMO VALUES ARE THE MEAN AUC OVER THE 21 HELD-OUT METHODS.
<table><tr><td>Protocol</td><td>rPPG+DCT</td><td>CLIP</td><td>Fusion</td></tr><tr><td>Random video split</td><td>0.680</td><td>0.976</td><td>0.974</td></tr><tr><td>LOMO, loose</td><td>0.929</td><td>0.994</td><td>0.996</td></tr><tr><td>LOMO, clean reals</td><td>0.600</td><td>0.971</td><td>0.967</td></tr><tr><td>LOMO, strict</td><td>0.634</td><td>0.947</td><td>0.929</td></tr><tr><td>Identity-disjoint, strict</td><td>0.680</td><td>0.957</td><td>0.938</td></tr></table>

## B. Custom protocols

Studies that build their own protocols from these datasets can introduce larger leaks. In a leave-one-method-out protocol where real test videos also appear in training, a weak rPPG+DCT detector scores 0.929 [0.925, 0.933]; holding identities out reduces this to 0.634 [0.585, 0.683], a difference of 0.30 (Table VIII). The same change costs CLIP 0.05 (0.994 to 0.947 [0.934, 0.959]). Weak features are inflated far more than strong ones, plausibly because memorized real videos are a larger share of what they can use. Holding out only the official test reals while sharing identities gives 0.600 [0.562, 0.641] for rPPG+DCT, so the inflation comes mainly from reusing test videos. The LOMO and identity-disjoint rows use a fixed 30 epochs; with early stopping the identitydisjoint values change by less than 0.01.

## VIII. DISCUSSION

Why animation may add flicker. A real video records one continuous scene, so its color changes slowly. A generator that animates a still image renders each frame from the same source appearance plus a predicted motion, and nothing ties one frame’s small color errors to the next. Frame-wide flicker, down to alternation between consecutive frames, is a natural result, and it fits the effect appearing when a sequence is synthesized from one frame but not when each frame is rebuilt from itself. We offer this as a hypothesis; testing it requires access to generator internals.

Implications for deployment. A detector that encodes video source will raise false alarms on genuine footage from unseen cameras, platforms, or processing chains, as the FF++ results show.

Recommended controls. Detectors motivated by physiology should report, alongside accuracy: (1) the same model on background regions and on full frames without face detection; (2) a frame-shuffle control; (3) separate performance in the heart-rate band and outside it; (4) results with real and fake videos sampled identically and re-encoded identically; (5) real-versus-real and cross-source scoring; and (6) identity-disjoint splits that account for every identity named in a filename.

Limitations. Without ground-truth pulse, we can show that the detectors do not rely primarily on a pulse, not that every fake lacks one. The test set has about 100 real videos from 9 identities, and the self-reenactment experiment uses two generators, 32 to 34 test clips per class, and a frame-by-frame control for one generator only. The real videos’ acquisition and processing history also contributes to the cue, and the near-Nyquist excess in TF fakes is much larger than in our self-reenactments, so part of it may come from generator output or encoding pipelines rather than from animation itself. Our trace classifiers share the design of published rPPG detectors but are not those systems; DeepFakesON-Phys, the one we could run, reaches 0.575 on TF. We do not compare against audio-visual or lip-sync detectors, since our question is what rPPG-style detectors learn. Trace detectors also transfer poorly across datasets (face CHROM 0.504 from TF to FF++).

## IX. CONCLUSION

On the forgeries where the physiological premise is strongest, rPPG-style detectors mostly do not detect a pulse. On talking-face forgeries, a plain chrominance trace reaches 0.998 AUC from the face, 0.998 from a background region, and 0.999 from the full frame, above a dedicated rPPG extractor (0.869). Its information lies above the heart-rate band (0.992 for 4–12 Hz alone vs. 0.943 for 0.7–4 Hz), the models rely on alternation between consecutive frames, and the signal disappears when frames are shuffled (0.521). Lowpassing fake traces at 8 Hz makes 97% of them appear real, adding fake-level 12–12.5 Hz content to real traces makes 99.8% appear fake, and injecting a synthetic pulse leaves only 2.5% of fakes called real. Animating a real video from its own first frame makes it separable from its source (up to 0.931), while rebuilding each frame from itself does not (0.582), which points to frame-to-frame color flicker introduced by image-to-video synthesis.

The same detectors also encode where a video comes from. A face chrominance detector trained on Celeb-DF++ labels 99.1% of real FF++ videos as fake; a frozen CLIP detector shows the same behavior and is not improved by adding rPPG and frequency features (0.957 alone vs. 0.938 fused). Evaluation practice can hide these problems: the official Celeb-DF splits share identities and source videos between training and test, and a leave-one-method-out protocol that reuses test reals in training inflates an rPPG+DCT detector from 0.634 to 0.929 AUC. Physiological detectors should be judged by controls that test their premise, not by accuracy alone.

## ETHICAL IMPACT STATEMENT

This work uses only public research datasets (Celeb-DF, Celeb-DF++, FaceForensics++) under their terms of use and collects no new data from people; the self-reenactments were generated from existing dataset videos for analysis only. Showing what detectors rely on could help evasion, but the controls we recommend make detectors without a valid premise easier to catch.

## ACKNOWLEDGMENT

This work was supported by a Google Cloud research credits grant.

## REFERENCES

[1] W. Verkruysse, L. O. Svaasand, and J. S. Nelson, “Remote plethysmographic imaging using ambient light,” Optics Express, vol. 16, no. 26, pp. 21434–21445, 2008.

[2] G. de Haan and V. Jeanne, “Robust pulse rate from chrominance-based rPPG,” IEEE Trans. Biomed. Eng., vol. 60, no. 10, pp. 2878–2886, 2013.

[3] W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, “Algorithmic principles of remote PPG,” IEEE Trans. Biomed. Eng., vol. 64, no. 7, pp. 1479–1491, 2017.

[4] B. Zou et al., “RhythmFormer: Extracting patterned rPPG signals based on periodic sparse attention,” Pattern Recognition, vol. 164, p. 111511, 2025.

[5] U. A. Ciftci, I. Demir, and L. Yin, “FakeCatcher: Detection of synthetic portrait videos using biological signals,” IEEE Trans. Pattern Anal. Mach. Intell., 2020.

[6] H. Qi et al., “DeepRhythm: Exposing DeepFakes with attentional visual heartbeat rhythms,” in Proc. ACM Multimedia, 2020.

[7] J. Hernandez-Ortega, R. Tolosana, J. Fierrez, and A. Morales, “DeepFakesON-Phys: DeepFakes detection based on heart rate estimation,” in AAAI Workshops, 2021.

[8] J. Liang and W. Deng, “Identifying rhythmic patterns for face forgery detection and categorization,” in Proc. IJCB, 2021.

[9] X. Li, J. Komulainen, G. Zhao, P.-C. Yuen, and M. Pietikäinen, “Generalized face anti-spoofing by detecting pulse from face videos,” in Proc. ICPR, 2016.

[10] Y. Liu, A. Jourabloo, and X. Liu, “Learning deep models for face anti-spoofing: Binary or auxiliary supervision,” in Proc. CVPR, 2018.

[11] O. Harraq and T. Aldwairi, “PulseGuard: rPPG-derived temporal signals as a forensic modality for talking-face deepfake detection,” arXiv:2607.21776, 2026.

[12] O. Harraq and T. Aldwairi, “Complementary rPPG-derived and lip-region frequency cues for talking-face deepfake detection,” arXiv:2609.22284, 2026.

[13] A. D’Amelio et al., “On using rPPG signals for DeepFake detection: A cautionary note,” in Proc. ICIAP, 2023.

[14] C. Seibold et al., “High-quality deepfakes have a heart!” Frontiers in Imaging, 2025.

[15] J. Hernandez-Ortega, J. Fierrez, A. Morales, and P. Tome, “Time analysis of pulse-based face anti-spoofing in visible and NIR,” in Proc. CVPR Workshops, 2018.

[16] A. Haliassos, K. Vougioukas, S. Petridis, and M. Pantic, “Lips don’t lie: A generalisable and robust approach to face forgery detection,” in Proc. CVPR, 2021.

[17] Y. Zheng, J. Bao, D. Chen, M. Zeng, and F. Wen, “Exploring temporal coherence for more general video face forgery detection,” in Proc. ICCV, 2021.

[18] Y. Yu, R. Ni, S. Yang, Y. Ni, Y. Zhao, and A. C. Kot, “Mining generalized multi-timescale inconsistency for detecting deepfake videos,” Int. J. Comput. Vis., 2025.

[19] Z. Guo, Y. Liu, J. Zhang, H. Zheng, and S. Shan, “Face forgery video detection via temporal forgery cue unraveling,” in Proc. CVPR, 2025.

[20] R. Geirhos et al., “Shortcut learning in deep neural networks,” Nature Machine Intelligence, vol. 2, pp. 665–673, 2020.

[21] S. Smeu et al., “Circumventing shortcuts in audio-visual deepfake detection datasets with unsupervised learning,” in Proc. CVPR, 2025.

[22] P. Grommelt et al., “Fake or JPEG? Revealing common biases in generated image detection datasets,” in Proc. ECCV Workshops, 2024.

[23] S. Dong, J. Wang, R. Ji, J. Liang, H. Fan, and Z. Ge, “Implicit identity leakage: The stumbling block to improving deepfake detection generalization,” in Proc. CVPR, 2023.

[24] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF: A large-scale challenging dataset for DeepFake forensics,” in Proc. CVPR, 2020.

[25] Y. Li, D. Zhu, X. Cui, and S. Lyu, “Celeb-DF++: A large-scale challenging video DeepFake benchmark for generalizable forensics,” arXiv:2507.18015, 2025.

[26] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “FaceForensics++: Learning to detect manipulated facial images,” in Proc. ICCV, 2019.

[27] V. Bazarevsky, Y. Kartynnik, A. Vakunov, K. Raveendran, and M. Grundmann, “BlazeFace: Sub-millisecond neural face detection on mobile GPUs,” arXiv:1907.05047, 2019.

[28] A. Radford et al., “Learning transferable visual models from natural language supervision,” in Proc. ICML, 2021.

[29] F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in Proc. CVPR, 2017.

[30] J. Zhao and H. Zhang, “Thin-plate spline motion model for image animation,” in Proc. CVPR, 2022.

[31] J. Guo, D. Zhang, X. Liu, Z. Zhong, Y. Zhang, P. Wan, and D. Zhang, “LivePortrait: Efficient portrait animation with stitching and retargeting control,” arXiv:2407.03168, 2024.